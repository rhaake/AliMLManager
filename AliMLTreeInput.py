# AliMLTreeInput: Create compatible data for the network from tree inputs
from __future__ import print_function
import numpy, math, logging, sys

import AliMLModels

#### GLOBAL SETTINGS
dummy = 0
kNumConstituents = 50

#__________________________________________________________________________________________________________
def GetGenerator():
  """Return user-implemented data generator"""
  return AliMLTreeDataGenerator()

#__________________________________________________________________________________________________________
def GetDataset(index):
  """Dataset in framework-readable format"""

  if index == 0:
    logging.info('Using test data set, 1 class loaded')

    # Define the parts of the dataset
    # Here we have 2 trees: One will contribute 60% to the full class, one 40%
    datasetA     = {'file': './Misc/testfile.root', 'treename': 'Jets_TypeA', 'weight': 0.6}
    datasetB     = {'file': './Misc/testfile.root', 'treename': 'Jets_TypeB', 'weight': 0.4}

    # Define dataset classes & their cuts
    # We only add one class, but could add more in case of a classification task
    classes = []
    classes.append({'datasets': [datasetA, datasetB], 'cuts': [{'var': 'JetPt', 'min': 15., 'max': 120.}, ]})
    return classes

  else:
    logging.error('Error: Dataset {:d} not defined!'.format(index))
    return []

#__________________________________________________________________________________________________________
def GetModel(scheme, loadModel, verbose=True):
  """Model definition"""
  # Instantiate and load model on-demand (here: regression model, 1 class)
  myModel = AliMLModels.AliMLKerasModel(1, GetGenerator()) 
  if loadModel:
    myModel.LoadModel(scheme)

  # Define demanded model
  if scheme == 'SimpleModel':
    if not loadModel:
      # The input types are defined within the data generator below
      myModel.AddBranchCNN1D([16,32,64], [2,0,0], 1, [2,2,2], 0.1, inputType='constituents', activation='tanh')
      myModel.SetFinalLayer(4, 128,0.25, 'ridge')

      myModel.fInit = 'he_uniform'
      myModel.fOptimizer = 'rmsprop'
      myModel.fActivation = 'tanh'
      myModel.fLossFunction = 'mean_squared_error'
      myModel.CreateModel(scheme)
      myModel.fBatchSize = 512
    myModel.fLearningRate = 0.00001
  else:
    logging.error('Error: Model scheme {:s} not defined!'.format(scheme))

  if verbose:
    myModel.PrintProperties()

  return myModel


#__________________________________________________________________________________________________________
class AliMLTreeDataGenerator:
  """Class to generate a userdefined data item as numpy array"""

  ###############################################
  def __init__ (self):
    self.fCache = {}

  ###############################################
  def CreateEmptyArray(self, dtype, numEntries):
    """Create the empty array with correct size according to requested type"""
    typeShape = (numEntries,) + self.GetDataTypeShape(dtype)
    data = numpy.zeros(shape=typeShape)

    return data

  ###############################################
  def GetSample(self, rawData, dtype, index):
    """Create a sample for a datatype using rawData
       rawData is a tree handle
    """
    if dtype == 'properties':
      data = ([rawData.JetPt, rawData.JetEta, rawData.JetPhi])
    elif dtype == 'jetpt':
      data = ([rawData.JetPt,])
    elif dtype == 'constituents':
      data = (self.GetJetConstituents(rawData))
    return data

  ###############################################
  def GetSampleTruth(self, rawData, classID=None):
    """Define what is considered as truth of sample"""
    return rawData.JetPt

  ###############################################
  def GetDataTypeShape(self, inputType):
    """Gives the data format for different given data types
       Must be the format that is defined in GetSample()
    """
    if inputType=='properties':
      return (3,)
    elif inputType=='jetpt':
      return (1,)
    elif inputType=='constituents':
      return (kNumConstituents, 3)
    else:
      raise ValueError('Input {:s} not valid.'.format(inputType))


  ###############################################
  def GetJetConstituents(self, rawData):
    """Return array of constituents"""
    jetConstituents = numpy.zeros(shape=(kNumConstituents, 3))

    for i in range(min(rawData.NumConst, kNumConstituents)):
      jetConstituents[i][0] = rawData.ConstPt[i]
      jetConstituents[i][1] = rawData.ConstEta[i]
      jetConstituents[i][2] = rawData.ConstPhi[i]

    return jetConstituents
