# AliMLAnalysisTools: Some helper functions for analysis
from __future__ import print_function
import numpy, logging, copy, array
import ROOT


###############################################
def CalculateTaggingRatesBinaryClassifier(model, scores, scores_ref, test_batch_size, eff=1.0, refEff=1.0, verbose=0):
  """Show and return classification efficiencies when demanding different efficiencies in data_ref"""
  scores_ref = copy.deepcopy(scores_ref)
  scores_ref = numpy.sort(scores_ref, axis=0)
  currentThresholds = {}
  for perc in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
    currentThresholds[perc] = scores_ref[int(float(perc)*(len(scores_ref)))][0]

  # Define the counters and check the scores
  tagged   = {'0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0, '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 0,'0.9': 0}
  for i in range(len(scores)):
    score = scores[i]
    for perc in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
      if score <= currentThresholds[perc]: # prediction is class 0
        tagged[perc] += 1

  if verbose == 1:
    for perc in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
      print('At efficiency {:.1f}% (abs: {:2.3f}%)(score < {:E}), tagging rate={:3.4f}%, absolute rate={:3.4f}%'.format(100.*float(perc), 100.*float(perc)*refEff, currentThresholds[perc], 100.*(float(tagged[perc])/float(len(scores))), 100.*(float(tagged[perc])/float(len(scores)))*eff))

  return ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],[float(tagged[perc])/len(scores) for perc in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']])


###############################################
def GenerateROCCurve(truth, score):
  """ROC curve & AUC are generated"""
  from sklearn.metrics import roc_curve, auc
  currentROCy, currentROCx, _ = roc_curve(truth, score)
  currentAUC = auc(currentROCy, currentROCx)
  print('AUC={:f}'.format(currentAUC))
  return (currentAUC, currentROCy, currentROCx)


###############################################
def GetThresholdsInBinsOfProperty(model, data, data_property, efficiencies, binning_property, test_batch_size):
  """Function to get score thresholds to achieve given efficiencies in bins of a property (e.g. pT)"""
  # ########
  # Properties:
  #   model:            network
  #   data:             data for the network
  #   data_property:    property that should be used for the binning
  #   efficiencies:     efficiencies for each property-bin
  #   binning_property: binning in property

  # Calculate the scores of the samples
  scores     = model.predict(data, batch_size=test_batch_size, verbose=0)

  ########## Calculate the ref. thresholds for each data bin
  thresholds = []
  for i in range(len(binning_property)):
    # Create a list of scores in the specified data range
    tmpScores = []
    for j,score in enumerate(scores):
      if data_property[j] >= binning_property[i] and data_property[j] < binning_property[i+1]:
        tmpScores.append(score[0])

    # In this list, find the threshold for the eff. given in efficiencies
    if len(tmpScores):
      tmpScores = numpy.sort(tmpScores, axis=0)
      currentThreshold = tmpScores[int(float(efficiencies[i])*(len(tmpScores)))]
      thresholds.append(currentThreshold)
    else:
      thresholds.append(0)

  return thresholds


###############################################
def AddBranchToTreeInFile(fname, tname, bname, customData):
  """Add custom branches to tree in file"""

  ##### Try to read file & tree
  ofile   = ROOT.TFile(fname, 'update')
  outTree = ofile.Get(tname)
  ##### Create tree, if it does not exist
  firstBranch = False
  if not outTree:
    firstBranch = True
    outTree = ROOT.TTree(tname, tname)

  custom  = numpy.zeros(1, dtype=float)
  brCustom = outTree.Branch(bname, custom, '{:s}/D'.format(bname))

  ##### Loop through the chain and add raw samples to output tree
  for iData, data in enumerate(customData):
    try:
      custom[0] = data[0]
    except:
      custom[0] = data

    if firstBranch:
      outTree.Fill()
    else:
      outTree.GetEntry(iData)
      brCustom.Fill()

  outTree.Write('', ROOT.TObject.kOverwrite)
  ofile.Close()
