#!/usr/bin/env python
from __future__ import print_function
import os, numpy, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no info output from tensorflow
import AliMLLoader, AliMLModels, AliMLHelpers, AliMLAnalysisTools, AliMLManager
import AliMLJets

numpy.random.seed(13377331)  # for reproducibility

#######################################################################################################
#### SETTINGS

gNumEpochs             = 50           # number of training epochs
gNumEventsTraining     = 200000       # training events per class
gNumEventsValidation   = 50000        # validation events per class
gEventChunkSize        = 200000       # event chunk size
gEventOffset           = 0            # offset after which the events are used for training


########## OPTIONS
gLoadModel             = True             # load model
gDoTraining            = False            # Train the model
gDoExtraction          = True             # Extract samples for further processing
gDoUserAnalysis        = False            # User analysis

# User defined schemes that will be used
gSchemes               = ['HQ_SimpleConstImpact_C']  # 'HQ_Simple_B', 'HQ_SimpleAll_B', 'HQ_SimpleConst_B', 'HQ_SimpleConstImpact_B', 'HQ_Simple_B' 'HQ_Simple_B', QG_Deep, 'HQ_Special_B'
gExtractionName        = 'HQC'


gUserModule            = AliMLJets
gDataGenerator         = gUserModule.GetGenerator()

gDataset               = 6  # 0/3,2,5,6

#######################################################################################################
def Run():
  """Main control function"""

  if gDoTraining or gDoUserAnalysis:
    for scheme in gSchemes:
      model = gUserModule.GetModel(scheme, gLoadModel)
      dataset      = gUserModule.GetDataset(gDataset)
      if gDoTraining:
        AliMLManager.DoTraining(model, dataset, gDataGenerator, gNumEpochs, gNumEventsTraining, gNumEventsValidation, gEventChunkSize, gEventOffset, cacheMode='off')
      if gDoUserAnalysis:
        DoUserAnalysis(model, dataset)

  # Extract data once (potentially for more than one model given in gSchemes
  dataset = gUserModule.GetDataset(gDataset)
  if gDoExtraction:
    #AddScoreToTree(gSchemes, dataset, extractTrainData=True)
    DoSampleExtraction(gSchemes, dataset, extractTrainData=False)


#######################################################################################################
def DoSampleExtraction(schemes, dataset_training, extractTrainData = True):
  """Extract samples to root file. 
     Extraction of the full tree should work regardless of the data input
  """
  extractTestData  = True # test data
  extractFullTree  = True  # tree samples instead of single quantities

  fileName = '{:s}Samples_{:s}_D{:d}'.format('Tree' if extractFullTree else '', gExtractionName, gDataset)
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
  if os.path.isfile('./Results/{:s}.root'.format(fileName)):
    os.remove('./Results/{:s}.root'.format(fileName))

  # Extract the used datasets, tree by tree
  for dset, cuts in zip(uniqueDatasets, uniqueDatasetsCuts):
    for iScheme,scheme in enumerate(schemes):
      model = gUserModule.GetModel(scheme, True, verbose=False)
      dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
      dataLoader.fVerbose = False
      dataLoader.fMaxProducedSamples = 100000000
      dataLoader.AddClass([dset,], cuts)

      nTestingEvents  = dataLoader.GetNumSamples()
      if iScheme==0:
        weight = 0.0
        for dclass in dataset_training:
          for dset_tr in dclass['datasets']:
            if dset_tr['file'] == dset['file'] and dset_tr['treename'] == dset['treename']:
              weight += dset_tr['weight']
        trainingEvents = weight*(gNumEventsTraining+gNumEventsValidation)
        testingEvents = nTestingEvents - trainingEvents
        logging.info('Found {:d} events in total for this dataset.'.format(nTestingEvents))
        logging.info('Extraction: {:.0f} training, {:.0f} testing.'.format(trainingEvents, testingEvents))


      if extractTrainData:
        # Fastforward all events that have been used in the training/validation datasets
        weight = 0.0
        for dclass in dataset_training:
          for dset_tr in dclass['datasets']:
            if dset_tr['file'] == dset['file'] and dset_tr['treename'] == dset['treename']:
              weight += dset_tr['weight']

        nTrainingEvents = weight*(gNumEventsTraining+gNumEventsValidation)
        nTestingEvents -= nTrainingEvents

        ############ Extract the data
        (data, _, _, rawDataIndicesTraining) = dataLoader.GetDataChunk(nTrainingEvents)

        ############ For the first scheme/model only, extract some properties (they are not different for the different models)
        if iScheme == 0:
          dataLoader.CreateTreeFromRawData('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), rawDataIndices=rawDataIndicesTraining)
          # Get list of pTs for the dataset
          chain = dataLoader.GetRawDataChain()
          pt, vtxMass, vtxSLxy, centrality = [], [], [], []
          for it in rawDataIndicesTraining:
            chain.GetEntry(it)

            pt.append(chain.Jets.Pt())
            centrality.append(chain.Jets.Centrality())

            # Find most displaced vertex
            mostDisplacedM = 0
            mostDisplacedL = -1
            mostDisplacedSL = 0
            for i in range(chain.Jets.GetNumbersOfSecVertices()):
              vtx = chain.Jets.GetSecondaryVertex(i)
              if vtx.Lxy() > mostDisplacedL:
                mostDisplacedM  = vtx.Mass()
                mostDisplacedSL = vtx.Lxy()/vtx.SigmaLxy();
                mostDisplacedL  = vtx.Lxy()
            vtxMass.append(mostDisplacedM)
            vtxSLxy.append(mostDisplacedSL)

          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), 'Pt', customData=pt)
          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), 'Centrality', customData=centrality)
          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), 'VtxMass', customData=vtxMass)
          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), 'VtxSLxy', customData=vtxSLxy)
          logging.info('Extracted training data')

        ############

        # Predict the scores
        scores    = model.fModel.predict(data, batch_size=512, verbose=0)
        AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), 'Score_{:s}'.format(model.fModelName), customData=scores)
        logging.info('Extracted training scores for model {:s}'.format(model.fModelName))
      else:
        pass 
        # TODO: Fast forward events

      if extractTestData:
        ############ Extract the data (testing)
        (data, _, _, rawDataIndicesTesting) = dataLoader.GetDataChunk(nTestingEvents)

        ############ For the first scheme/model only, extract some properties (they are not different for the different models)
        if iScheme == 0:
          dataLoader.CreateTreeFromRawData('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), rawDataIndices=rawDataIndicesTesting)
          # Get list of pTs for the dataset
          chain = dataLoader.GetRawDataChain()
          pt, vtxMass, vtxSLxy, centrality = [], [], [], []
          for it in rawDataIndicesTesting:
            chain.GetEntry(it)

            pt.append(chain.Jets.Pt())
            centrality.append(chain.Jets.Centrality())

            # Find most displaced vertex
            mostDisplacedM = 0
            mostDisplacedL = -1
            mostDisplacedSL = 0
            for i in range(chain.Jets.GetNumbersOfSecVertices()):
              vtx = chain.Jets.GetSecondaryVertex(i)
              if vtx.Lxy() > mostDisplacedL:
                mostDisplacedM  = vtx.Mass()
                mostDisplacedSL = vtx.Lxy()/vtx.SigmaLxy();
                mostDisplacedL  = vtx.Lxy()
            vtxMass.append(mostDisplacedM)
            vtxSLxy.append(mostDisplacedSL)

          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), 'Pt', customData=pt)
          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), 'Centrality', customData=centrality)
          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), 'VtxMass', customData=vtxMass)
          AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), 'VtxSLxy', customData=vtxSLxy)
          logging.info('Extracted testing data')

        ############

        # Predict the scores
        scores    = model.fModel.predict(data, batch_size=512, verbose=0)
        AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), 'Score_{:s}'.format(model.fModelName), customData=scores)
        logging.info('Extracted testing scores for model {:s}'.format(model.fModelName))

    logging.info('Tree extraction done.')


#######################################################################################################
def AddScoreToTree(schemes, dataset_training, extractTrainData = True):
  extractTestData  = True # test data

  fileName = '{:s}Samples_{:s}_D{:d}'.format('Tree', gExtractionName, gDataset)
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

  # Add score from used datasets, tree by tree
  for dset, cuts in zip(uniqueDatasets, uniqueDatasetsCuts):
    for iScheme,scheme in enumerate(schemes):
      model = gUserModule.GetModel(scheme, True, verbose=False)
      dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
      dataLoader.fVerbose = False
      dataLoader.fMaxProducedSamples = 100000000
      dataLoader.AddClass([dset,], cuts)

      nTestingEvents  = dataLoader.GetNumSamples()
      if iScheme==0:
        weight = 0.0
        for dclass in dataset_training:
          for dset_tr in dclass['datasets']:
            if dset_tr['file'] == dset['file'] and dset_tr['treename'] == dset['treename']:
              weight += dset_tr['weight']
        trainingEvents = weight*(gNumEventsTraining+gNumEventsValidation)
        testingEvents = nTestingEvents - trainingEvents
        logging.info('Found {:d} events in total for this dataset.'.format(nTestingEvents))
        logging.info('Extraction: {:.0f} training, {:.0f} testing.'.format(trainingEvents, testingEvents))


      if extractTrainData:
        # Fastforward all events that have been used in the training/validation datasets
        weight = 0.0
        for dclass in dataset_training:
          for dset_tr in dclass['datasets']:
            if dset_tr['file'] == dset['file'] and dset_tr['treename'] == dset['treename']:
              weight += dset_tr['weight']

        nTrainingEvents = weight*(gNumEventsTraining+gNumEventsValidation)
        nTestingEvents -= nTrainingEvents

        ############ Extract the data
        (data, _, _, rawDataIndicesTraining) = dataLoader.GetDataChunk(nTrainingEvents)

        # Predict the scores
        scores    = model.fModel.predict(data, batch_size=512, verbose=0)
        AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Training_{:s}'.format(dset['treename']), 'Score_{:s}'.format(model.fModelName), customData=scores)
        logging.info('Extracted training scores for model {:s}'.format(model.fModelName))
      else:
        pass 
        # TODO: Fast forward events

      if extractTestData:
        ############ Extract the data (testing)
        (data, _, _, rawDataIndicesTesting) = dataLoader.GetDataChunk(nTestingEvents)

        # Predict the scores
        scores    = model.fModel.predict(data, batch_size=512, verbose=0)
        AliMLAnalysisTools.AddBranchToTreeInFile('./Results/{:s}.root'.format(fileName), 'Testing_{:s}'.format(dset['treename']), 'Score_{:s}'.format(model.fModelName), customData=scores)
        logging.info('Extracted testing scores for model {:s}'.format(model.fModelName))

    logging.info('Tree extraction done.')


#######################################################################################################
def DoUserAnalysis(model, dataset):
  """Place analysis etc. here"""

  if False:
    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dataset[0]['datasets'][0],], dataset[0]['cuts'])
    dataLoader.FastForward(gNumEventsTraining)
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(gNumEventsValidation + gNumEventsTraining)
    scores_c     = model.fModel.predict(data, batch_size=512, verbose=0)

    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dataset[1]['datasets'][0],], dataset[0]['cuts'])
    dataLoader.FastForward(gNumEventsTraining*0.9)
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(gNumEventsValidation*0.9 + gNumEventsTraining)
    scores_l     = model.fModel.predict(data, batch_size=512, verbose=0)

    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dataset[1]['datasets'][1],], dataset[0]['cuts'])
    dataLoader.FastForward(gNumEventsTraining*0.1)
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(gNumEventsValidation*0.1 + gNumEventsTraining)
    scores_b     = model.fModel.predict(data, batch_size=512, verbose=0)

    print('Mistagging rates for b-jets in c-tagging:')
    AliMLAnalysisTools.CalculateTaggingRatesBinaryClassifier(model, scores_b, scores_c, 512, eff=0.03, refEff=0.06, verbose=1)
    print('Mistagging rates for udsg-jets in c-tagging:')
    AliMLAnalysisTools.CalculateTaggingRatesBinaryClassifier(model, scores_l, scores_c, 512, eff=0.91, refEff=0.06, verbose=1)

  if True:
    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dataset[0]['datasets'][0],], dataset[0]['cuts'])
    dataLoader.FastForward(gNumEventsTraining*1.0)
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(gNumEventsValidation*1 + gNumEventsTraining)
    scores_b     = model.fModel.predict(data, batch_size=512, verbose=0)

    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dataset[1]['datasets'][1],], dataset[1]['cuts'])
    dataLoader.FastForward(gNumEventsTraining*0.1)
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(gNumEventsValidation*0.1 + gNumEventsTraining)
    scores_c     = model.fModel.predict(data, batch_size=512, verbose=0)

    dataLoader = AliMLLoader.AliMLDataLoader(gDataGenerator, model.fRequestedData, 1)
    dataLoader.fMaxProducedSamples = 100000000
    dataLoader.AddClass([dataset[1]['datasets'][0],], dataset[1]['cuts'])
    dataLoader.FastForward(gNumEventsTraining*0.9)
    (data, _, _, rawDataIndices) = dataLoader.GetDataChunk(gNumEventsValidation*0.9 + gNumEventsTraining)
    scores_l     = model.fModel.predict(data, batch_size=512, verbose=0)


    print('Mistagging rates for c-jets in b-tagging:')
    AliMLAnalysisTools.CalculateTaggingRatesBinaryClassifier(model, scores_c, scores_b, 512, eff=0.06, refEff=0.03, verbose=1)
    print('Mistagging rates for udsg-jets in b-tagging:')
    AliMLAnalysisTools.CalculateTaggingRatesBinaryClassifier(model, scores_l, scores_b, 512, eff=0.91, refEff=0.03, verbose=1)


#######################################################################################################

try:
  Run()
except:
  AliMLHelpers.NotifyError()
  raise
else:
  AliMLHelpers.NotifyDone()
