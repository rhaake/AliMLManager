#  results.py

import numpy, logging
import ROOT
import AliMLHelpers, AliMLAnalysisTools

class AliMLModelResultsBase:
  """Base class that holds model results and perform tests on it"""
  ###############################################
  def __init__(self, name):
    self.fModelName = name

    # Arrays holding values obtained for each learning step
    self.fHistoryLossTraining       = []
    self.fHistoryLossValidation     = []
    self.fHistoryLearningRate       = []
    self.fHistoryNumberEvents       = []


  ###############################################
  def GetReachedLoss(self):
    return self.fHistoryLossValidation[len(self.fHistoryLossValidation)-1]


  ###############################################
  def AddResult(self, loss_train, loss_val, lr, numEvents):
    """Add performance results from a model training"""
    self.fHistoryLossTraining.append(loss_train)
    self.fHistoryLossValidation.append(loss_val)
    self.fHistoryLearningRate.append(lr)

    # In case, the result contains more than one epoch, create the list of numEvents
    offset = 0 if len(self.fHistoryNumberEvents) == 0 else self.fHistoryNumberEvents[len(self.fHistoryNumberEvents)-1]
    self.fHistoryNumberEvents.append(offset+numEvents)


  ###############################################
  def CreatePlots(self):
    """This function saves the plots using the existing results data"""
    ##### Save histograms/plots
    epochs    = range(len(self.fHistoryNumberEvents))
    # Loss
    AliMLHelpers.SavePlot('./Results/{:s}-Loss.png'.format(self.fModelName), 'Loss function', x=epochs, y=(self.fHistoryLossTraining, self.fHistoryLossValidation), rangey=(0.8* min(self.fHistoryLossValidation),1.2*max(self.fHistoryLossValidation)), functionlabels=('Training', 'Validation'))


#__________________________________________________________________________________________________________
class AliMLModelResultsBinClassifier(AliMLModelResultsBase):
  """Model result class for binary classifier results"""

  ###############################################
  def __init__(self, name):
    AliMLModelResultsBase.__init__(self, name)

    self.fCurrentAUC                = None
    self.fCurrentROCx               = None
    self.fCurrentROCy               = None
    self.fCurrentTestScores         = None
    # Values for classification
    self.fHistoryAccuracyTraining   = []
    self.fHistoryAccuracyValidation = []
    self.fHistoryAUC                = []


  ###############################################
  def GetReachedAccuracy(self):
    return self.fHistoryAccuracyValidation[len(self.fHistoryAccuracyValidation)-1]
  def GetReachedAUC(self):
    return self.fHistoryAUC[len(self.fHistoryAUC)-1]


  ###############################################
  def AddResult(self, loss_train, loss_val, lr, numEvents, accuracy_train, accuracy_val, model, data, truth, mergedData, mergedTruth, test_batch_size):
    """Add performance results from a model training"""
    AliMLModelResultsBase.AddResult(self, loss_train, loss_val, lr, numEvents)
    self.fHistoryAccuracyTraining.append(accuracy_train)
    self.fHistoryAccuracyValidation.append(accuracy_val)

    if not model:
      raise ValueError('Cannot test a model that was not correctly created.')

    # Evaluate scores on dataset
    self.fCurrentTestScores = [None,] * 2
    for i in range(2):
      self.fCurrentTestScores[i]  = model.predict( [data[i][j] for j in range(len(mergedData))], batch_size=test_batch_size, verbose=0)

    # Calculate mistagging rates
    (percentages, rates) = AliMLAnalysisTools.CalculateTaggingRatesBinaryClassifier(model, self.fCurrentTestScores[1], self.fCurrentTestScores[0], test_batch_size)
    for perc, rate in zip(percentages, rates):
      print('At efficiency {:.1f}%, mistagging rate={:3.3f}%'.format(100.*perc, 100.*rate))

    # Calculate ROC/ROC curve
    merged_score  = model.predict(mergedData, batch_size=test_batch_size, verbose=0)
    (self.fCurrentAUC, self.fCurrentROCy, self.fCurrentROCx) = AliMLAnalysisTools.GenerateROCCurve(mergedTruth, merged_score[:,0])
    self.fHistoryAUC.append(self.fCurrentAUC)

    self.CreatePlots()

  ###############################################
  def CreatePlots(self):
    """This function saves the plots using the existing results data"""
    AliMLModelResultsBase.CreatePlots(self)

    # Test dataset and evaluate scores
    labels = ['Class 1', 'Class 2']

    ##### Save histograms/plots
    epochs    = range(len(self.fHistoryNumberEvents))

    # Accuracy
    AliMLHelpers.SavePlot('./Results/{:s}-Accuracy.png'.format(self.fModelName), 'Accuracy function', x=epochs, y=(self.fHistoryAccuracyTraining, self.fHistoryAccuracyValidation), functionlabels=('Training', 'Validation'), legendloc='lower right')

    # Score
    AliMLHelpers.SaveHistogram('./Results/{:s}-Scores.png'.format(self.fModelName), 'Scores on validation data', tuple(self.fCurrentTestScores), tuple(labels), rangex=(0,1), logY=True)
    # AUC
    AliMLHelpers.SavePlot('./Results/{:s}-AUC.png'.format(self.fModelName), 'AUC values', x=self.fHistoryNumberEvents, y=(self.fHistoryAUC,), functionlabels=('AUC',), legendloc='lower right')
    # ROC
    AliMLHelpers.SavePlot('./Results/{:s}-ROC.png'.format(self.fModelName), 'ROC curve', x=self.fCurrentROCy, y=(self.fCurrentROCx,self.fCurrentROCy), functionlabels=('(AUC={0:.3f})'.format(self.fCurrentAUC),'Guess ROC'), rangex=(0,1.1), legendloc='lower right', axislabels=('False Positive Rate', 'True Positive Rate') )


#__________________________________________________________________________________________________________
class AliMLModelResultsRegressionTask(AliMLModelResultsBase):
  """Model result class for regression task results"""

  ###############################################
  def __init__(self, name):
    AliMLModelResultsBase.__init__(self, name)

    self.fHistoryStdDeviation               = []
    self.fHistoryMeanDeviation              = []
    self.fHistoryMeanAbsDeviation           = []
    self.fHistoryMeanRelDeviation           = []

  ###############################################
  def AddResult(self, loss_train, loss_val, lr, numEvents, accuracy_train, accuracy_val, model, data, truth, mergedData, mergedTruth, test_batch_size):
    """Add performance results from a model training"""
    AliMLModelResultsBase.AddResult(self, loss_train, loss_val, lr, numEvents)

    if not model:
      raise ValueError('Cannot test a model that was not correctly created.')

    predictions = model.predict(mergedData, batch_size=test_batch_size)
    deviation = 0
    absDeviation = 0
    relDeviation = 0
    sumOfSquares = 0
    for i in range(len(predictions)):
      predicted = predictions[i][0]
      trueValue = mergedTruth[i]
      deviation    += (trueValue - predicted)
      absDeviation += abs(trueValue - predicted)
      relDeviation += (trueValue - predicted)/trueValue if trueValue else 0
      sumOfSquares += (trueValue - predicted) * (trueValue - predicted)

    # In case of more than one epochs for this result, add the corresponding number of history entries
    for i in range(len(loss_train)):
      self.fHistoryStdDeviation.append(1./len(predictions) * math.sqrt(sumOfSquares))
      self.fHistoryMeanDeviation.append(1./len(predictions) * deviation)
      self.fHistoryMeanRelDeviation.append(1./len(predictions) * relDeviation)
      self.fHistoryMeanAbsDeviation.append(1./len(predictions) * absDeviation)

    self.CreatePlots()

  ###############################################
  def CreatePlots(self):
    """This function saves the plots using the existing results data"""
    AliMLModelResultsBase.CreatePlots(self)

    ##### Save histograms/plots
    epochs    = range(len(self.fHistoryNumberEvents))
    # Mean deviation
    AliMLHelpers.SavePlot('./Results/{:s}-MeanDeviation.png'.format(self.fModelName), 'MeanDeviation', x=epochs, y=(self.fHistoryMeanDeviation,), functionlabels=('Mean deviation',), legendloc='lower right')
    AliMLHelpers.SavePlot('./Results/{:s}-MeanAbsDeviation.png'.format(self.fModelName), 'MeanAbsDeviation', x=epochs, y=(self.fHistoryMeanAbsDeviation,), functionlabels=('Mean abs. deviation',), legendloc='lower right')
    AliMLHelpers.SavePlot('./Results/{:s}-MeanRelDeviation.png'.format(self.fModelName), 'MeanRelDeviation', x=epochs, y=(self.fHistoryMeanRelDeviation,), functionlabels=('Mean rel. deviation',), legendloc='lower right')
    AliMLHelpers.SavePlot('./Results/{:s}-StdDeviation.png'.format(self.fModelName), 'StdDeviation', x=epochs, y=(self.fHistoryStdDeviation,), functionlabels=('Std. deviation',), legendloc='lower right')

