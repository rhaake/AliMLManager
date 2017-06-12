#!/usr/bin/env python
from __future__ import print_function
import os, numpy, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no info output from tensorflow
import AliMLLoader, AliMLModels, AliMLHelpers, AliMLAnalysisTools, AliMLManager
import AliMLTreeInput

numpy.random.seed(7331)  # for reproducibility

#######################################################################################################
#### SETTINGS

gNumEpochs             = 50           # number of training epochs
gNumEventsTraining     = 50000        # training events per class
gNumEventsValidation   = 10000        # validation events per class
gEventChunkSize        = 50000        # event chunk size
gEventOffset           = 0            # offset after which the events are used for training


########## OPTIONS
gLoadModel             = False            # load model
gDoTraining            = True             # Train the model
gDoUserAnalysis        = False            # User analysis

# User defined schemes that will be used
gSchemes               = ['SimpleModel'] 


gUserModule            = AliMLTreeInput
gDataGenerator         = gUserModule.GetGenerator()

gDataset               = 0

#######################################################################################################
def Run():
  """Main control function"""

  if gDoTraining or gDoUserAnalysis:
    for scheme in gSchemes:
      model = gUserModule.GetModel(scheme, gLoadModel)
      dataset      = gUserModule.GetDataset(gDataset)
      if gDoTraining:
        AliMLManager.DoTraining(model, dataset, gDataGenerator, gNumEpochs, gNumEventsTraining, gNumEventsValidation, gEventChunkSize, gEventOffset)
      if gDoUserAnalysis:
        DoUserAnalysis(model, dataset)


#######################################################################################################
def DoUserAnalysis(model, dataset):
  """Place analysis etc. here"""
  logging.info('User-defined analysis done.')


#######################################################################################################

try:
  Run()
except:
  AliMLHelpers.NotifyError()
  raise
else:
  AliMLHelpers.NotifyDone()
