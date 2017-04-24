# AliMLJets: Creates jet properties from AliBasicJet objects
from __future__ import print_function

from   keras.utils.np_utils import to_categorical
import numpy, math, logging, sys
import AliMLModels


#### GLOBAL SETTINGS
kNumConstituents = 30
kNumSecondaryVtx = 50
kNumConstituentParameters = 3
kNumHistogramBins = 100
kImageResolution = 65
kImageR = 0.3
kImageDrawCorrelationLine = False
kImageDrawJTLine = False
kImageUseL2Norm = True

gDataset_bJets     = None
gDataset_cJets     = None
gDataset_lightJets = None


#__________________________________________________________________________________________________________
def GetGenerator():
  """Return user-implemented data generator"""
  return AliMLJetDataGenerator()

#__________________________________________________________________________________________________________
def GetDataset(index):
  """Dataset in framework-readable format"""

  if index == 0:
    logging.info('Using b-jets vs. c-/udsg-jets (20/80) dataset (15-120 GeV/c)')

    # Define the parts of the dataset
    gDataset_bJets     = {'file': '/opt/alice/MachineLearning/Data/LHC14g3a.root', 'treename': 'bJets', 'weight': 1.0}
    gDataset_cJets     = {'file': '/opt/alice/MachineLearning/Data/LHC14g3a.root', 'treename': 'cJets', 'weight': 0.1}
    gDataset_lightJets = {'file': '/opt/alice/MachineLearning/Data/LHC14g3b.root', 'treename': 'lightJets', 'weight': 0.9}


    # Define dataset classes & their cuts
    classes = []
    classes.append({'datasets': [gDataset_bJets], 'cuts': [{'branch': 'Jets', 'var': 'Pt', 'min': 15., 'max': 120.}, ]})
    classes.append({'datasets': [gDataset_lightJets, gDataset_cJets], 'cuts': [{'branch': 'Jets', 'var': 'Pt', 'min': 15., 'max': 120.}, ]})
    return classes

  else:
    logging.error('Error: Dataset {:d} not defined!'.format(index))
    return []

#__________________________________________________________________________________________________________
def GetModel(scheme, loadModel):
  """Model definition"""
  # Instantiate and load model on-demand
  myModel = AliMLModels.AliMLKerasModel(2, GetGenerator()) 
  if loadModel:
    myModel.LoadModel(scheme)

  # Define demanded model
  if scheme == 'HQ_Deep_B':
    if not loadModel:
      myModel.AddBranchDense(4,128,0.1,inputType='shapes')
      myModel.AddBranchCNN1D([128,64,64], [2,0,2], 1, [4,2,2], 0.1, inputType='constituents_impactchargedjt')
      myModel.AddBranchCNN1D([128,64,64], [2,0,2], 1, [4,2,2], 0.1, inputType='constituents_etaphir')
      myModel.AddBranchCNN1D([64,128,256], [2,0,0], 1, [8,4,2], 0.1, inputType='secVtx_novtxmass')
      myModel.SetFinalLayer(4, 128,0.25)

      myModel.fInit = 'he_uniform'
      myModel.fOptimizer = 'adam'
      myModel.fLossFunction = 'binary_crossentropy'
      myModel.CreateModel(scheme)
      myModel.fBatchSize = 512
      myModel.fLearningRate = 0.0001
  elif scheme == 'HQ_Shallow_FC':
    if not loadModel:
      myModel.AddBranchDense(4, 64, 0.1, inputType='secVtx_significantlxy')
      myModel.AddBranchDense(4, 64, 0.1, inputType='secVtx_dispersion')
      myModel.AddBranchDense(4, 64, 0.1, inputType='secVtx_chi2')
      myModel.AddBranchDense(4, 64, 0.1, inputType='secVtx_l')
      myModel.SetFinalLayer(4, 128,0.25)
      myModel.fInit = 'he_uniform'
      myModel.fOptimizer = 'adam'
      myModel.fLossFunction = 'binary_crossentropy'
      myModel.CreateModel(scheme)
      myModel.fBatchSize = 512
      myModel.fLearningRate = 0.0001
  else:
    logging.error('Error: Model scheme {:s} not defined!'.format(scheme))

  myModel.PrintProperties()
  return myModel


#__________________________________________________________________________________________________________
class AliMLJetDataGenerator:
  """Class to generate a userdefined jet data item as numpy array"""

  ###############################################
  def __init__ (self):
    # Type of truth extracted for each datasample
    self.fTruthSelection = 'classes'
    self.fCache = {}

  ###############################################
  def CreateEmptyArray(self, dtype, numEntries):
    """Create the empty array with correct size according to requested type"""
    typeShape = (numEntries,) + self.GetDataTypeShape(dtype)
    data = numpy.zeros(shape=typeShape)

    return data

  ###############################################
  def GetSample(self, rawData, dtype, index):
    """Create a sample for a datatype using rawData"""
    cacheTag = 'samplecache_{:s}'.format(dtype)

    # Check whether we have a new sample
    # if yes, empty the cache
    if 'samplecache_index' not in self.fCache:
      self.fCache['samplecache_index'] = index
    elif self.fCache['samplecache_index'] != index:
      self.fCache = {}

    # Check if the requested type is cached
    # if not, calculate it
    if cacheTag in self.fCache:
      data = self.fCache[cacheTag]
    else:
      rawDataJet = rawData.Jets
      if dtype == 'images':
        data = (self.GetJetImage(rawDataJet))
      elif dtype == 'chargedimages':
        data = (self.GetJetImageWithCharge(rawDataJet))
      elif dtype == 'shapes':
        data = ([rawDataJet.Pt(), self.GetJetShape('Mass', rawData), self.GetJetShape('Dispersion', rawData), self.GetJetShape('Radial', rawData), self.GetJetShape('LeSub', rawData), self.GetJetShape('ConstPtDeviation', rawData), rawDataJet.GetNumbersOfConstituents()])
      elif dtype == 'constituents':
        data = (self.GetJetConstituents(0, rawDataJet))
      elif dtype == 'constituents_ptjt':
        data = (self.GetJetConstituents(1, rawDataJet))
      elif dtype == 'constituents_etaphir':
        data = (self.GetJetConstituents(2, rawDataJet))
      elif dtype == 'constituents_ptr':
        data = (self.GetJetConstituents(3, rawDataJet))
      elif dtype == 'constituents_etaphijt':
        data = (self.GetJetConstituents(4, rawDataJet))
      elif dtype == 'constituents_impactjt':
        data = (self.GetJetConstituents(5, rawDataJet))
      elif dtype == 'constituents_pidjtr':
        data = (self.GetJetConstituents(6, rawDataJet))
      elif dtype == 'constituents_impactchargedjt':
        data = (self.GetJetConstituents(7, rawDataJet))
      elif dtype == 'histogrampt':
        data = (self.GetHistogramJetConstituent('pt', rawDataJet))
      elif dtype == 'histogramr':
        data = (self.GetHistogramJetConstituent('r', rawDataJet))
      elif dtype == 'histogramjt':
        data = (self.GetHistogramJetConstituent('jt', rawDataJet))
      elif dtype == 'jetprop':
        data = ([rawDataJet.Pt(), rawDataJet.Area(), rawDataJet.GetNumbersOfConstituents()])
      elif dtype == 'secVtx':
        data = (self.GetSecondaryVertices(rawDataJet, useVtxMass=True))
      elif dtype == 'secVtx_novtxmass':
        data = (self.GetSecondaryVertices(rawDataJet, useVtxMass=False))
      elif dtype in ['secVtx_lxy', 'secVtx_sigmaxy', 'secVtx_dispersion', 'secVtx_chi2', 'secVtx_l', 'secVtx_significantlxy']:
        data = (self.GetSecondaryVerticesProperty(rawDataJet, ['secVtx_lxy', 'secVtx_sigmaxy', 'secVtx_dispersion', 'secVtx_chi2', 'secVtx_l', 'secVtx_significantlxy'].index(dtype)))
      elif dtype in ['secVtx_lxy_2D', 'secVtx_sigmaxy_2D', 'secVtx_dispersion_2D', 'secVtx_chi2_2D', 'secVtx_l_2D', 'secVtx_significantlxy_2D']:
        data = (self.GetSecondaryVerticesProperty(rawDataJet, ['secVtx_lxy_2D', 'secVtx_sigmaxy_2D', 'secVtx_dispersion_2D', 'secVtx_chi2_2D', 'secVtx_l_2D', 'secVtx_significantlxy_2D'].index(dtype), in2D=True))
      elif dtype == 'production_vertices':
        data = (self.GetProductionVertices(rawDataJet))
      # Save data to cache and return it
      self.fCache[cacheTag] = data

    return data

  ###############################################
  def GetSampleTruth(self, rawData, classID=None):
    """Define what is considered as truth of sample"""
    
    if self.fTruthSelection == 'classes':
      jet_truth = classID
    elif self.fTruthSelection == 'PID':
      jet_truth = rawData.Jets.MotherHadronMatching()
    elif self.fTruthSelection == 'Bgrd':
      jet_truth = rawData.Jets.BackgroundDensity()
    else:
      raise ValueError('Truth type {} not implemented yet.'.format(self.fTruthSelection))

    return jet_truth

  ###############################################
  def GetDataTypeShape(self, inputType):
    """Gives the data format for different given data types"""
    if inputType=='constituents':
      return (kNumConstituents, 3)
    elif inputType=='constituents_ptjt':
      return (kNumConstituents, 2)
    elif inputType=='constituents_etaphir':
      return (kNumConstituents, 3)
    elif inputType=='constituents_ptr':
      return (kNumConstituents, 2)
    elif inputType=='constituents_etaphijt':
      return (kNumConstituents, 3)
    elif inputType=='constituents_impactjt':
      return (kNumConstituents, 3)
    elif inputType=='constituents_impactchargedjt':
      return (kNumConstituents, 3)
    elif inputType=='constituents_pidjtr':
      return (kNumConstituents, 6)
    elif inputType=='images':
      return (1, kImageResolution, kImageResolution)
    elif inputType=='chargedimages':
      return (2, kImageResolution, kImageResolution)
    elif inputType=='histogramr':
      return (kNumHistogramBins,)
    elif inputType=='histogrampt':
      return (kNumHistogramBins,)
    elif inputType=='histogramjt':
      return (kNumHistogramBins,)
    elif inputType=='shapes':
      return (7,)
    elif inputType=='jetprop':
      return (3,)
    elif inputType=='secVtx':
      return (kNumSecondaryVtx, 9)
    elif inputType=='secVtx_novtxmass':
      return (kNumSecondaryVtx, 8)
    elif inputType in ['secVtx_lxy', 'secVtx_sigmaxy', 'secVtx_dispersion', 'secVtx_chi2', 'secVtx_l', 'secVtx_significantlxy']:
      return (kNumSecondaryVtx,)
    elif inputType in ['secVtx_lxy_2D', 'secVtx_sigmaxy_2D', 'secVtx_dispersion_2D', 'secVtx_chi2_2D', 'secVtx_l_2D', 'secVtx_significantlxy_2D']:
      return (kNumSecondaryVtx, 1)
    elif inputType=='production_vertices':
      return (kNumConstituents, 3)
    else:
      raise ValueError('Input {:s} not valid.'.format(inputType))


  ###############################################
  def GetHistogramJetConstituent(self, mode, jet):
    histogram = numpy.zeros(shape=kNumHistogramBins)
    if mode == 'pt':
      for i in range(jet.GetNumbersOfConstituents()):
        histogram[int(round(jet.GetJetConstituent(i).Pt()))] += 1
    elif mode == 'r' or mode == 'jt':
      for i in range(jet.GetNumbersOfConstituents()):
        deltaEta = jet.Eta() - jet.GetJetConstituent(i).Eta();
        deltaPhi = min(abs(jet.Phi() - jet.GetJetConstituent(i).Phi()), math.pi*2.0 - abs(jet.Phi() - jet.GetJetConstituent(i).Phi()));
        deltaPhi = deltaPhi if (jet.Phi() - jet.GetJetConstituent(i).Phi()) < 0 and (jet.Phi() - jet.GetJetConstituent(i).Phi()) <= math.pi else -deltaPhi
        deltaR = math.sqrt(deltaPhi*deltaPhi + deltaEta*deltaEta)

        if mode == 'r':
          if int(round(deltaR*100)) < 100:
            histogram[int(round(deltaR*100))] += 1
        elif mode == 'jt':
          jt     = jet.GetJetConstituent(i).Pt() * math.sin(deltaR)
          if int(round(jt*2)) < 100:
            histogram[int(round(jt*2))] += 1

    return histogram

  ###############################################
  def GetJetShape(self, stype, data):
    return getattr(data, 'fJetShape{:s}'.format(stype))

  ###############################################
  def GetJetImage(self, jet, angle=0):
    jet_image = numpy.zeros(shape=(kImageResolution,kImageResolution))

    # Correct jet
    correctionAngle = self._GetRotationAngleForCenteredJet(jet)

    px_last_x = -1
    px_last_y = -1
    pt_last = -1
    totalIntensity = 0
    # Now go through all correlations and draw them as lines into the matrix
    # Sort jet with pT
    jetConsts = sorted([jet.GetJetConstituent(i) for i in range(jet.GetNumbersOfConstituents())], key=lambda k: k.Pt(), reverse=True)
    for j in range(0, len(jetConsts)):
      deltaEta = (jet.Eta() - jetConsts[j].Eta())
      deltaPhi = min(abs(jet.Phi() - jetConsts[j].Phi()),math.pi*2.0 - abs(jet.Phi() - jetConsts[j].Phi()));
      deltaPhi = deltaPhi if (jet.Phi() - jetConsts[j].Phi()) < 0 and (jet.Phi() - jetConsts[j].Phi()) <= math.pi else -deltaPhi
      pt = (jetConsts[j].Pt())
      # Rotate the constituent around center
      (deltaEta, deltaPhi) = self._GetRotatedEtaPhi(deltaEta, deltaPhi, angle+correctionAngle)
      if abs(deltaEta) >= kImageR or abs(deltaPhi) >= kImageR:
        continue
      # Pixel coordinates
      centerPixel = kImageResolution//2 # jet-axis
      # Scaled such that the image is from -0.5..+0.5 in eta/phi
      px_x = centerPixel + (kImageResolution-1)*deltaEta/(kImageR*2)
      px_y = centerPixel + (kImageResolution-1)*deltaPhi/(kImageR*2)

      jet_image[int(px_x),int(px_y)] += pt
      totalIntensity += pt*pt

      # OPTIONAL: Draw line between jet constituents and jet axis
      if kImageDrawJTLine:
        rows, columns = line(int(px_x), int(px_y), centerPixel, centerPixel)
        for pix in range(1,len(rows)-1):
          val = pt * math.sin(math.sqrt(deltaPhi*deltaPhi + deltaEta*deltaEta)) # constituent jt
          jet_image[rows[pix],columns[pix]] += val
          totalIntensity += val*val


      # OPTIONAL: Draw correlation line between jet constituents
      if kImageDrawCorrelationLine and px_last_x >= 0:
        rows, columns = line(int(px_x), int(px_y), int(px_last_x), int(px_last_y))
        for pix in range(1,len(rows)-1):
          val = (pt-pt_last) # *(float(pix)/len(rows))
          jet_image[rows[pix],columns[pix]] += val
          totalIntensity += val*val

      # Save last pixel
      px_last_x = px_x
      px_last_y = px_y
      pt_last = pt

    if kImageUseL2Norm and totalIntensity:
      jet_image /= totalIntensity

    return jet_image


  ###############################################
  def GetJetImageWithCharge(self, jet, angle=0):
    jet_image = numpy.zeros(shape=(2,kImageResolution,kImageResolution))

    # Correct jet
    correctionAngle = self._GetRotationAngleForCenteredJet(jet)

    px_last_x = -1
    px_last_y = -1
    pt_last = -1
    totalIntensity = 0
    # Now go through all correlations and draw them as lines into the matrix
    # Sort jet with pT
    jetConsts = sorted([jet.GetJetConstituent(i) for i in range(jet.GetNumbersOfConstituents())], key=lambda k: k.Pt(), reverse=True)
    for j in range(0, jet.GetNumbersOfConstituents()):
      deltaEta = (jet.Eta() - jetConsts[j].Eta())
      deltaPhi = min(abs(jet.Phi()- jetConsts[j].Phi()),math.pi*2.0 - abs(jet.Phi() - jetConsts[j].Phi()));
      deltaPhi = deltaPhi if (jet.Phi() - jetConsts[j].Phi()) < 0 and (jet.Phi() - jetConsts[j].Phi()) <= math.pi else -deltaPhi
      pt = (jetConsts[j].Pt())
      # Rotate the constituent around center
      (deltaEta, deltaPhi) = self._GetRotatedEtaPhi(deltaEta, deltaPhi, angle+correctionAngle)
      if abs(deltaEta) >= kImageR or abs(deltaPhi) >= kImageR:
        continue
      # Pixel coordinates
      centerPixel = kImageResolution//2 # jet-axis
      # Scaled such that the image is from -0.5..+0.5 in eta/phi
      px_x = centerPixel + (kImageResolution-1)*deltaEta/(kImageR*2)
      px_y = centerPixel + (kImageResolution-1)*deltaPhi/(kImageR*2)

      if jetConsts[j].Charge() >= 0:
        jet_image[0][int(px_x)][int(px_y)] += pt
      elif jetConsts[j].Charge() < 0:
        jet_image[1][int(px_x)][int(px_y)] += pt

      totalIntensity += pt*pt

      # OPTIONAL: Draw line between jet constituents and jet axis
      if kImageDrawJTLine:
        rows, columns = line(int(px_x), int(px_y), centerPixel, centerPixel)
        for pix in range(1,len(rows)-1):
          val = pt * math.sin(math.sqrt(deltaPhi*deltaPhi + deltaEta*deltaEta)) # constituent jt
          jet_image[rows[pix],columns[pix]] += val

          if jetConsts[j].Charge() >= 0:
            jet_image[0][rows[pix],columns[pix]] += pt
          elif jetConsts[j].Charge() < 0:
            jet_image[1][rows[pix],columns[pix]] += pt
          totalIntensity += val*val


      # OPTIONAL: Draw correlation line between jet constituents
      if kImageDrawCorrelationLine and px_last_x >= 0:
        rows, columns = line(int(px_x), int(px_y), int(px_last_x), int(px_last_y))
        for pix in range(1,len(rows)-1):
          val = (pt-pt_last) # *(float(pix)/len(rows))
          jet_image[rows[pix],columns[pix]] += val

          if jetConsts[j].Charge() >= 0:
            jet_image[0][rows[pix],columns[pix]] += pt
          elif jetConsts[j].Charge() < 0:
            jet_image[1][rows[pix],columns[pix]] += pt
          totalIntensity += val*val

      # Save last pixel
      px_last_x = px_x
      px_last_y = px_y
      pt_last = pt

    if kImageUseL2Norm and totalIntensity:
      jet_image /= totalIntensity

    return jet_image

  ###############################################
  def GetJetConstituents(self, mode, jet):

    # Create lists depending on mode
    if mode == 0:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 3))
    elif mode == 1:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 2)) # pt,jt
    elif mode == 2:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 3)) # eta,phi,R
    elif mode == 3:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 2)) # pt,R
    elif mode == 4:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 3)) # eta,phi,jt
    elif mode == 5:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 3)) # impact parameters, jt
    elif mode == 6:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 6)) # 4 pid signals, jt ,r
    elif mode == 7:
      jet_constituents = numpy.zeros(shape=(kNumConstituents, 3)) # impact parameters, charge sign * jt

    ##### Get sorted constituents and their quantities if not already cached
    constList = [jet.GetJetConstituent(i) for i in range(jet.GetNumbersOfConstituents())]
   
    if ('jetConsts' not in self.fCache.keys()):
      # Sort constituents with ascending pT
      self.fCache['jetConsts'] = sorted(constList, key=lambda k: k.Pt(), reverse=True)

      ##### Create meta quantities for constituents
      self.fCache['jetConsts.deltaEta'] = numpy.empty(shape=(jet.GetNumbersOfConstituents(),))
      self.fCache['jetConsts.deltaPhi'] = numpy.empty(shape=(jet.GetNumbersOfConstituents(),))
      self.fCache['jetConsts.deltaR']   = numpy.empty(shape=(jet.GetNumbersOfConstituents(),))
      self.fCache['jetConsts.jt']       = numpy.empty(shape=(jet.GetNumbersOfConstituents(),))

      for j in range(0, min(kNumConstituents, jet.GetNumbersOfConstituents())):
        self.fCache['jetConsts.deltaEta'][j] = (jet.Eta() - self.fCache['jetConsts'][j].Eta())
        self.fCache['jetConsts.deltaPhi'][j] = min(abs(jet.Phi()- self.fCache['jetConsts'][j].Phi()),math.pi*2.0 - abs(jet.Phi() - self.fCache['jetConsts'][j].Phi()));
        self.fCache['jetConsts.deltaPhi'][j] = self.fCache['jetConsts.deltaPhi'][j] if (jet.Phi() - self.fCache['jetConsts'][j].Phi()) < 0 and (jet.Phi() - self.fCache['jetConsts'][j].Phi()) <= math.pi else -self.fCache['jetConsts.deltaPhi'][j]
        self.fCache['jetConsts.deltaR'][j]   = math.sqrt(self.fCache['jetConsts.deltaPhi'][j]*self.fCache['jetConsts.deltaPhi'][j] + self.fCache['jetConsts.deltaEta'][j]*self.fCache['jetConsts.deltaEta'][j]) # dR
        self.fCache['jetConsts.jt'][j]       = self.fCache['jetConsts'][j].Pt() * math.sin(self.fCache['jetConsts.deltaR'][j]) # jt

    ##### Go through list of constituents
    for j in range(0, min(kNumConstituents, jet.GetNumbersOfConstituents())):

      if mode == 0:
        jet_constituents[j][0] = jet.GetJetConstituent(j).Pt() # pt
        jet_constituents[j][1] = self.fCache['jetConsts.deltaEta'][j]
        jet_constituents[j][2] = self.fCache['jetConsts.deltaPhi'][j]
      elif mode == 1:
        jet_constituents[j][0] = jet.GetJetConstituent(j).Pt() # pt
        jet_constituents[j][1] = self.fCache['jetConsts.jt'][j]
      elif mode == 2:
        jet_constituents[j][0] = self.fCache['jetConsts.deltaEta'][j]
        jet_constituents[j][1] = self.fCache['jetConsts.deltaPhi'][j]
        jet_constituents[j][2] = self.fCache['jetConsts.deltaR'][j] # R
      elif mode == 3:
        jet_constituents[j][0] = jet.GetJetConstituent(j).Pt() # pt
        jet_constituents[j][1] = self.fCache['jetConsts.deltaR'][j] # R
      elif mode == 4:
        jet_constituents[j][0] = self.fCache['jetConsts.deltaEta'][j]
        jet_constituents[j][1] = self.fCache['jetConsts.deltaPhi'][j]
        jet_constituents[j][2] = self.fCache['jetConsts.jt'][j]
      elif mode == 5:
        jet_constituents[j][0] = jet.GetJetConstituent(j).ImpactParameterD()
        jet_constituents[j][1] = jet.GetJetConstituent(j).ImpactParameterZ()
        jet_constituents[j][2] = self.fCache['jetConsts.jt'][j]
      elif mode == 6:
        jet_constituents[j][0] = self.fCache['jetConsts.jt'][j]
        jet_constituents[j][1] = self.fCache['jetConsts.deltaR'][j] # R
        jet_constituents[j][2] = jet.GetJetConstituent(j).PID().SignalITS()
        jet_constituents[j][3] = jet.GetJetConstituent(j).PID().SignalTPC()
        jet_constituents[j][4] = jet.GetJetConstituent(j).PID().SignalTOF()
        jet_constituents[j][5] = jet.GetJetConstituent(j).PID().SignalTRD()
      elif mode == 7:
        jet_constituents[j][0] = jet.GetJetConstituent(j).ImpactParameterD()
        jet_constituents[j][1] = jet.GetJetConstituent(j).ImpactParameterZ()
        jet_constituents[j][2] = jet.GetJetConstituent(j).Charge() * self.fCache['jetConsts.jt'][j]

    return jet_constituents


  ###############################################
  def GetListSecondaryVertices(self, jet):

    # Fill the internal sec vtx. objects if not already cached
    if 'secondaryVertices' not in self.fCache:
      # Go through the full list of secondary vertices and select the most significant ones
      secVtx = sorted([jet.GetSecondaryVertex(i) for i in range(jet.GetNumbersOfSecVertices())], key=lambda k: k.Dispersion(), reverse=False)
      secondary_vertices = numpy.zeros(shape=(kNumSecondaryVtx,9))
      acceptedVtx = 0
      for j in range(0, min(kNumSecondaryVtx, jet.GetNumbersOfSecVertices())):
        if secVtx[j].Chi2() <= 1e-12:
          continue
        if acceptedVtx >= kNumSecondaryVtx:
          break
        secondary_vertices[acceptedVtx][0] = secVtx[j].Vx() - jet.VertexX()
        secondary_vertices[acceptedVtx][1] = secVtx[j].Vy() - jet.VertexY()
        secondary_vertices[acceptedVtx][2] = secVtx[j].Vz() - jet.VertexZ()
        secondary_vertices[acceptedVtx][3] = secVtx[j].Lxy()
        secondary_vertices[acceptedVtx][4] = secVtx[j].Chi2()
        secondary_vertices[acceptedVtx][5] = secVtx[j].Dispersion()
        secondary_vertices[acceptedVtx][6] = secVtx[j].SigmaLxy()
        secondary_vertices[acceptedVtx][7] = math.sqrt((secVtx[j].Vx() - jet.VertexX())*(secVtx[j].Vx() - jet.VertexX()) + (secVtx[j].Vy() - jet.VertexY())*(secVtx[j].Vy() - jet.VertexY()) + (secVtx[j].Vz() - jet.VertexZ())*(secVtx[j].Vz() - jet.VertexZ()))
        secondary_vertices[acceptedVtx][8] = secVtx[j].Mass()

        acceptedVtx += 1

      self.fCache['secondaryVertices'] = secondary_vertices

    return self.fCache['secondaryVertices']

  ###############################################
  def GetSecondaryVertices(self, jet, useVtxMass):
    if useVtxMass:
      return self.GetListSecondaryVertices(jet)
    else:
      secondary_vertices = numpy.zeros(shape=(kNumSecondaryVtx, 8))
      vtxList = self.GetListSecondaryVertices(jet)
      for i,vtx in enumerate(vtxList):
        secondary_vertices[i] = vtxList[i][0:8]

      return secondary_vertices

  ###############################################
  def GetSecondaryVerticesProperty(self, jet, propertyID, in2D=False):
    if in2D:
      vtxProp = numpy.zeros(shape=(kNumSecondaryVtx, 1))
    else:
      vtxProp = numpy.zeros(shape=(kNumSecondaryVtx))

    secVertices = self.GetListSecondaryVertices(jet)
    propInd = [3, 6, 5, 4, 7] # Lxy, sigmaLxy, dispersion, chi2, L

    for i,vtx in enumerate(secVertices):
      if in2D:
        if propertyID == 5:
          vtxProp[i][0] = vtx[3]/vtx[6] if vtx[6] else 0
        else:
          vtxProp[i][0] = vtx[propInd[propertyID]]
      else:
        if propertyID == 5:
          vtxProp[i] = vtx[3]/vtx[6] if vtx[6] else 0
        else:
          vtxProp[i] = vtx[propInd[propertyID]]

    return vtxProp

  ###############################################
  def GetProductionVertices(self, jet):
    secondary_vertices = numpy.zeros(shape=(kNumConstituents,3))
    acceptedVtx = 0

    for j in range(0, min(kNumConstituents, jet.GetNumbersOfConstituents())):

      # Cut a certain fraction of vertices (~20%)
      if numpy.random.uniform() > 1-0.2:#1-0.02:
        continue

      # Don't use secondary vertices more than 10 cm away from interaction point
      if abs(jet.GetJetConstituent(j).Vx()) >= 10 or abs(jet.GetJetConstituent(j).Vy()) >= 10 or abs(jet.GetJetConstituent(j).Vz()) >= 10:
        continue

      secondary_vertices[acceptedVtx][0] = jet.GetJetConstituent(j).Vx()
      secondary_vertices[acceptedVtx][1] = jet.GetJetConstituent(j).Vy()
      secondary_vertices[acceptedVtx][2] = jet.GetJetConstituent(j).Vz()

      if True:
        # Fluctuate assuming a resolution of 120 um (gaussian)
        # source for resolution: http://aliceinfo.cern.ch/ITS/sites/aliceinfo.cern.ch.ITS/files/documents/posterQM_ITSUpgrade.pdf
        secondary_vertices[acceptedVtx][0] = numpy.random.normal(secondary_vertices[acceptedVtx][0], scale=0.012)
        secondary_vertices[acceptedVtx][1] = numpy.random.normal(secondary_vertices[acceptedVtx][1], scale=0.012)
        secondary_vertices[acceptedVtx][2] = numpy.random.normal(secondary_vertices[acceptedVtx][2], scale=0.012)

        # Cut to the correct precision
        secondary_vertices[acceptedVtx][0] = round(secondary_vertices[acceptedVtx][0], 3)
        secondary_vertices[acceptedVtx][1] = round(secondary_vertices[acceptedVtx][1], 3)
        secondary_vertices[acceptedVtx][2] = round(secondary_vertices[acceptedVtx][2], 3)

      acceptedVtx += 1

    return secondary_vertices


  ###############################################
  def _GetRotationAngleForCenteredJet(self, jet):
    cog_x = 0.0
    cog_y = 0.0
    for j in range(0, jet.GetNumbersOfConstituents()):
      # Pixel properties
      deltaEta = (jet.Eta() - jet.GetJetConstituent(j).Eta())
      deltaPhi = min(abs(jet.Phi()- jet.GetJetConstituent(j).Phi()),math.pi*2.0 - abs(jet.Phi() - jet.GetJetConstituent(j).Phi()));
      deltaPhi = deltaPhi if (jet.Phi() - jet.GetJetConstituent(j).Phi()) < 0 and (jet.Phi() - jet.GetJetConstituent(j).Phi()) <= math.pi else -deltaPhi
      pt = (jet.GetJetConstituent(j).Pt())

      cog_x += deltaEta
      cog_y += deltaPhi

    # Center of gravity of the jet
    cog_x /= jet.GetNumbersOfConstituents()
    cog_y /= jet.GetNumbersOfConstituents()

    # Rotate jet such that cog at the same position
    angle = math.atan2(cog_x, cog_y) + math.pi/2
    #print('Center of gravity: ({:1.4f},{:1.4f}), rotate with {:1.4f}'.format(cog_x, cog_y, math.degrees(angle)))

    return angle

  ###############################################
  def _GetRotationAngleForCenteredJetSecVtx(self, jet):
    cog_x = 0.0
    cog_y = 0.0
    entries = 0
    for j in range(0, len(jet['SecVertices'])):
      if jet['SecVertices'][j].Chi2() <= 1e-12:
        continue
      # Pixel properties
      deltaX = (jet.VertexX() - jet['SecVertices'][j].Vx())
      deltaY = (jet.VertexY() - jet['SecVertices'][j].Vy())

      if abs(deltaX) >= 0.05 or abs(deltaY) >= 0.05:
        continue

      cog_x += deltaX
      cog_y += deltaY
      entries += 1

    # Center of gravity of the jet
    if entries:
      cog_x /= entries
      cog_y /= entries

    # Rotate jet such that cog at the same position
    angle = math.atan2(cog_x, cog_y) + math.pi/2
    #print('Center of gravity: ({:1.4f},{:1.4f}), rotate with {:1.4f}'.format(cog_x, cog_y, math.degrees(angle)))

    return angle

  ###############################################
  def _GetRotatedEtaPhi(self, eta, phi, angle):
    # Use rotation matrix to transform
    newEta = eta*math.cos(angle) - phi*math.sin(angle)
    newPhi = eta*math.sin(angle) + phi*math.cos(angle)
    return (newEta, newPhi)
