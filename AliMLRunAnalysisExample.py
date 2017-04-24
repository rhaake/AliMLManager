#!/usr/bin/env python
from __future__ import print_function
import os, numpy, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no info output from tensorflow
import AliMLLoader, AliMLModels, AliMLHelpers, AliMLAnalysisTools, AliMLManager
import AliMLJets

numpy.random.seed(13377331)  # for reproducibility

#######################################################################################################
#### SETTINGS

gNumEpochs             = 100         # number of training epochs
gNumEventsTraining     = 200000      # training events per class
gNumEventsValidation   = 50000      # validation events per class
gEventChunkSize        = 200000       # event chunk size
gEventOffset           = 0          # offset after which the events are used for training


########## OPTIONS
gLoadModel             = False        # load model
gDoTraining            = True         # Train the model
gDoExtraction          = False        # Extract samples for further processing
gDoUserAnalysis        = False        # User analysis

# User defined schemes that will be used
gSchemes               = ['HQ_Deep_B']

gUserModule            = AliMLJets
gDataGenerator         = gUserModule.GetGenerator()

gDataset               = 0

#######################################################################################################
def Run():
  """Main control function"""

  for scheme in gSchemes:
    model = gUserModule.GetModel(scheme, gLoadModel)
    dataset      = gUserModule.GetDataset(gDataset)
    if gDoTraining:
      AliMLManager.DoTraining(model, dataset, gDataGenerator, gNumEpochs, gNumEventsTraining, gNumEventsValidation, gEventChunkSize, gEventOffset)
    if gDoExtraction:
      DoSampleExtraction(model, dataset)
    if gDoUserAnalysis:
      DoUserAnalysis(model, dataset)


#######################################################################################################
def DoSampleExtraction(model, dataset_training):
  """Extract samples to root file. 
     Extraction of the full tree should work regardless of the data input
  """

  ###############
  # Extraction w/ scores

  # Find all used datasets
  uniqueDatasets = []
  uniqueDatasetsCuts = []
  for dclass in dataset_training:
    for dset in dclass['datasets']:
      dictSet = {'file': dset['file'], 'treename': dset['treename'], 'weight': 1.0}
      if dictSet not in uniqueDatasets:
        uniqueDatasets.append(dictSet)
        uniqueDatasetsCuts.append(dclass['cuts'])

  # Delete output files before extraction
  for myFile in ['./Results/ScoredSamples.root', './Results/ScoredPt.root']:
    if os.path.isfile(myFile):
      os.remove(myFile)

  extractFullTree = False
  # Extract the used datasets, tree by tree
  for dset, cuts in zip(uniqueDatasets, uniqueDatasetsCuts):
    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dset,], cuts)

    # Fastforward all events that have been used in the training/validation datasets
    weight = 0.0
    for dclass in dataset_training:
      for dset_tr in dclass['datasets']:
        if dset_tr['file'] == dset['file'] and dset_tr['treename'] == dset['treename']:
          weight += dset_tr['weight']

    nTrainingEvents = weight*(gNumEventsTraining+gNumEventsValidation)
    nTestingEvents  = dataLoader.GetNumSamples()-nTrainingEvents

    # Extract training dataset
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(nTrainingEvents)
    scores     = model.fModel.predict(data, batch_size=512, verbose=0)
    if extractFullTree:
      dataLoader.CreateTreeFromRawData('c', 'Training_{:s}'.format(dset['treename']), scores=scores, rawDataIndices=rawDataIndices)
    else:
      # Get list of pTs for the dataset
      chain = dataLoader.GetRawDataChain()
      it, pt = 0, []
      while chain.GetEntry(it):
        it += 1
        pt.append(chain.Jets.TruePt())
        if it == len(scores):
          break

      # Add pt's and scores to a tree
      AliMLAnalysisTools.AddBranchToTreeInFile('./Results/ScoredPt.root', 'Training_{:s}'.format(dset['treename']), 'Pt', customData=pt)
      AliMLAnalysisTools.AddBranchToTreeInFile('./Results/ScoredPt.root', 'Training_{:s}'.format(dset['treename']), 'Score', customData=scores)
 

    # Extract testing dataset
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(nTestingEvents)
    scores     = model.fModel.predict(data, batch_size=512, verbose=0)
    if extractFullTree:
      dataLoader.CreateTreeFromRawData('./Results/ScoredSamples.root', 'Testing_{:s}'.format(dset['treename']), scores=scores, rawDataIndices=rawDataIndices)
    else:
      # Get list of pTs for the dataset
      chain = dataLoader.GetRawDataChain()
      it, pt = 0, []
      while chain.GetEntry(it):
        it += 1
        pt.append(chain.Jets.TruePt())
        if it == len(scores):
          break

      AliMLAnalysisTools.AddBranchToTreeInFile('./Results/ScoredPt.root', 'Testing_{:s}'.format(dset['treename']), 'Pt', customData=pt)
      AliMLAnalysisTools.AddBranchToTreeInFile('./Results/ScoredPt.root', 'Testing_{:s}'.format(dset['treename']), 'Score', customData=scores)


#######################################################################################################
def DoUserAnalysis(model, dataset):
  """Place analysis etc. here"""
  pass

#######################################################################################################

try:
  Run()
except:
  AliMLHelpers.NotifyError()
  raise
else:
  AliMLHelpers.NotifyDone()
