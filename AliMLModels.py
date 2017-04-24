import keras
from keras.layers import LSTM, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten, Conv2D, MaxPooling2D, SimpleRNN, LocallyConnected1D, LocallyConnected2D, ZeroPadding2D, Input
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import History, EarlyStopping
from keras.regularizers import l2
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
import keras.backend as K

import sys, os, math, numpy, cPickle, logging
from AliMLResults import *


#__________________________________________________________________________________________________________
class CallbackSaveModel(keras.callbacks.Callback):
  """Keras callback that call the SaveModel function after each epoch"""
  def __init__(self, saveFunction, model, valData, valTruth, valDataMerged, valTruthMerged, batch_size, results, nevents, dovalidation):
    super(CallbackSaveModel, self).__init__()
    self.fSaveFunction = saveFunction
    self.fModel = model
    self.fValidationData  = valData
    self.fValidationTruth = valTruth
    self.fMergedValidationData  = valDataMerged
    self.fMergedValidationTruth = valTruthMerged
    self.fBatchSize = batch_size
    self.fResults = results
    self.fNEvents = nevents
    self.fDoValidation = dovalidation
  def on_epoch_end(self, epoch, logs=None):
    # Save the model on each epoch's end
    self.fSaveFunction()
    if self.fDoValidation:
      learningRate = K.get_value(self.fModel.optimizer.lr)
      self.fResults.AddResult(logs.get('loss'), logs.get('val_loss'), learningRate, self.fNEvents, logs.get('acc'), logs.get('val_acc'), self.fModel, self.fValidationData, self.fValidationTruth, self.fMergedValidationData, self.fMergedValidationTruth, self.fBatchSize)

class EnhancedProgbarLogger(keras.callbacks.ProgbarLogger):
  """Custom progress bar"""
  def on_train_begin(self, logs=None):
    keras.callbacks.ProgbarLogger.on_train_begin(self, logs)
    self.verbose = 1
  def on_epoch_begin(self, epoch, logs=None):
    print('')
    logging.info('')
    keras.callbacks.ProgbarLogger.on_epoch_begin(self, epoch, logs)

#__________________________________________________________________________________________________________
class AliMLKerasModel:
  """Keras model meta class"""
  ###############################################
  def __init__(self, numClasses, generator):
    self.fModel = None
    # Temp branches used in building the model
    self.fTempModelBranches = []
    self.fTempModelBranchesInput  = []
    # Humanreadable properties of the network
    self.fModelBranchesInput  = []
    self.fModelBranchesOutput = []
    # Final layer properties (top vanilla network)
    self.fFinalLayerStructure = []
    self.fFinalLayerNumLayers = 3
    self.fFinalLayerNumNeuronsPerLayer = 512
    self.fFinalLayerDropout = 0.5
    self.fFinalLayerActivation = 'relu'
    # Model properties
    self.fOptimizer = 'SGD'
    self.fLossFunction = 'binary_crossentropy'
    self.fModelName = 'NonameMetaModel'
    self.fLearningRate = 0.05
    self.fInit = 'he_normal'
    self.fNumClasses = numClasses
    self.fBatchSize = 512
    # Misc
    self.fPerEpochValidation = True # do validation per epoch
    self.fShowModelSummary = True
    self.fRequestedData = []
    self.fDataGenerator = generator
    self.fResults = None

  ###############################################
  def PrintProperties(self):
    epochs = len(self.fResults.fHistoryLearningRate)
    # General information on the model
    if epochs > 0:
      print('\nLoss={:4.3f}, Acc={:4.3f}, AUC={:4.3f} after {:3d} epochs (lr={:6.5f}). optimizer={:s}, loss function={:s}, init={:s}'.format(self.fResults.GetReachedLoss(), self.fResults.GetReachedAccuracy(), self.fResults.GetReachedAUC(), epochs, self.fResults.fHistoryLearningRate[epochs-1], self.fOptimizer, self.fLossFunction, self.fInit))
    print('\n###########################\nUsing {:s}. Model branches:'.format(self.fModelName))
    for inBranch in self.fModelBranchesInput:
      print('  {}'.format(inBranch))
    print('###########################\n')

  ###############################################
  def GetOutputLayer(self, numClasses):
    if numClasses == 1: # regression task
      return Dense(1, activation='linear')
    elif numClasses == 2: # bin classification task
      return Dense(1, activation='sigmoid', kernel_initializer='he_normal')
    elif numClasses > 2: # classification task
      return Dense(numClasses, activation='sigmoid', kernel_initializer='he_normal')

  ###############################################
  def GetResultsObject(self, numClasses, name):
    if numClasses == 1:
      results = AliMLModelResultsRegressionTask(name)
    elif numClasses == 2:
      results = AliMLModelResultsBinClassifier(name)
    elif numClasses > 2:
      raise ValueError('Num classes > 2 not supported yet.') #TODO: What is missing? Results class only?

    return results

  ###############################################
  def CreateModel(self, mname):
    if self.fModel:
      raise ValueError('Model already exists. Create a new one instead.')

    if not len(self.fTempModelBranches):
      raise ValueError('No branches given. Add some branches.')

    self.fModelName = mname

    # Merge the model branches to one model (that's the output we want to train)
    if len(self.fTempModelBranches) > 1:
      modelOutput = concatenate([self.fTempModelBranches[i] for i in range(len(self.fTempModelBranches))], name='ConcatenateLayer')
    else:
      modelOutput = self.fTempModelBranches[0]

    # Add final vanilla dense layer
    if self.fFinalLayerStructure == []:
      for i in range(self.fFinalLayerNumLayers):
        modelOutput = Dense(self.fFinalLayerNumNeuronsPerLayer, activation=self.fFinalLayerActivation, kernel_initializer=self.fInit, kernel_regularizer=l2(0.01), name='Final_{:d}_{:d}_activation_{:s}_regL2_0.01'.format(i, self.fFinalLayerNumNeuronsPerLayer, self.fFinalLayerActivation))(modelOutput)

        if self.fFinalLayerDropout:
          modelOutput = Dropout(self.fFinalLayerDropout, name='Final_{:d}_{:3.2f}'.format(i, self.fFinalLayerDropout))(modelOutput)
    else:
      for i, nNodes in enumerate(self.fFinalLayerStructure):
        modelOutput = Dense(nNodes, activation=self.fFinalLayerActivation, kernel_initializer=self.fInit, kernel_regularizer=l2(0.01), name='Final_{:d}_{:d}_activation_{:s}_regL2_0.01'.format(i, self.fFinalLayerNumNeuronsPerLayer, self.fFinalLayerActivation))(modelOutput)

        if self.fFinalLayerDropout:
          modelOutput = Dropout(self.fFinalLayerDropout, name='Final_{:d}_{:3.2f}'.format(i, self.fFinalLayerDropout))(modelOutput)


    # Add final output layer
    modelOutput = self.GetOutputLayer(self.fNumClasses)(modelOutput)
    # Create model proxy
    self.fModel = Model(inputs=[inBranch for inBranch in self.fTempModelBranchesInput], outputs=modelOutput)

    ### *COMPILE*
    self.fModel.compile(loss=self.fLossFunction, optimizer=self.fOptimizer, metrics=["accuracy"])

    if self.fShowModelSummary:
      self.fModel.summary()
      plot_model(self.fModel, to_file='./Models/{:s}.png'.format(mname))

    self.fResults = self.GetResultsObject(self.fNumClasses, self.fModelName)

    return self.fModel


  ###############################################
  def TrainModel(self, data, truth, validationData, validationTruth, mergedValidationData, mergedValidationTruth, model=None, results=None, numEpochs=1):
    callbacks = []
    callbacks.append(EnhancedProgbarLogger())
    # Early stopping
    #callbacks.append(EarlyStopping(monitor='val_loss', patience=4))
    # Learning rate reduction on plateau
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, verbose=0))

    if model == None:
      model = self.fModel

    if results == None:
      results = self.fResults

    callbacks.append(CallbackSaveModel(self.SaveModel, model, validationData, validationTruth, mergedValidationData, mergedValidationTruth, self.fBatchSize, results, len(data[0]), self.fPerEpochValidation))

    if not model:
      raise ValueError('Cannot train a model that was not correctly created.')

    ###############
    # Set learning should work for theano & tensorflow
    K.set_value(model.optimizer.lr, self.fLearningRate)

    # Train and test for numEpochs epochs
    hist = (model.fit([data[i] for i in range(len(data))], truth, epochs=numEpochs, batch_size=self.fBatchSize,
                  validation_data=([mergedValidationData[i] for i in range(len(mergedValidationData))], mergedValidationTruth), verbose=0, callbacks=callbacks))

  ###############################################
  def SaveModel(self):
    saveObj = {}
    # Prevent saving of the keras model with pickle
    # Save keras model separately
    for key in self.__dict__:
      if key == 'fModel':
        if self.fModel:
          self.fModel.save('./Models/{:s}.h5'.format(self.fModelName))
      elif key == 'fDataGenerator':
        pass
      elif key.startswith('fTemp'):
        pass
      elif key == 'fOptimizer' and not isinstance(self.fOptimizer, basestring):
        saveObj[key] = self.fOptimizer.__class__.__name__
      else:
        saveObj[key] = self.__dict__[key]

    cPickle.dump(saveObj, open('./Models/{:s}.p'.format(self.fModelName), 'wb'))

  ###############################################
  def LoadModel(self, fname, meta_data_only=False):
    if not os.path.isfile('./Models/{:s}.p'.format(fname)) or not os.path.isfile('./Models/{:s}.h5'.format(fname)):
       raise ValueError('Error: Cannot load model due to missing files!')

    self.__dict__ = cPickle.load(open('./Models/{:s}.p'.format(fname), 'rb'))
    if not meta_data_only:
      self.fModel = load_model('./Models/{:s}.h5'.format(fname))
    self.fModelName = fname

  ###############################################
  def ExportModel(self):
    """Export model to json & weights-only h5 file. Usable in LWTNN"""
    self.fModel.save_weights('./Models/{:s}.weights.h5'.format(self.fModelName), overwrite=True)

    # Save network architecture to file
    jsonfile = open('./Models/{:s}.json', 'w')
    jsonfile.write(self.fModel.to_json())
    jsonfile.close()


  ###############################################
  def AddBranchDense(self, nLayers, nNeuronsPerLayer, dropout, inputType, activation='relu'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    # Create fully-connected layers
    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_FC_{:s}'.format(branchID, inputType))
    for i in range(nLayers):
      if i==0:
        model = Dense(nNeuronsPerLayer, activation=activation, kernel_initializer=self.fInit, name='B{:d}_FC_{:d}_{:d}_activation_{:s}'.format(branchID, i, nNeuronsPerLayer, activation, inputType))(inputLayer)
      else:
        model = Dense(nNeuronsPerLayer, activation=activation, kernel_initializer=self.fInit, name='B{:d}_FC_{:d}_{:d}_activation_{:s}'.format(branchID, i, nNeuronsPerLayer, activation, inputType))(model)

      if dropout:
        model = Dropout(dropout, name='B{:d}_FC_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)

    self.fModelBranchesOutput.append('B{:d}_FC_{:d}_{:3.2f}'.format(branchID, i, dropout))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def AddBranchDenseCustom(self, layers, dropout, inputType, activation='relu'):
    self.fRequestedData.append(inputType)
    branchID = len(self.TempfModelBranches)

    # Create fully-connected layers
    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_FC_{:s}'.format(branchID, inputType))
    for i in range(len(layers)):
      if i==0:
        model = Dense(layers[i], activation=activation, kernel_initializer=self.fInit, name='B{:d}_FC_{:d}_{:d}_activation_{:s}'.format(branchID, i, layers[i], activation, inputType))(inputLayer)
      else:
        model = Dense(layers[i], activation=activation, kernel_initializer=self.fInit, name='B{:d}_FC_{:d}_{:d}_activation_{:s}'.format(branchID, i, layers[i], activation, inputType))(model)

      if dropout:
        model = Dropout(dropout, name='B{:d}_FC_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)

    self.fModelBranchesOutput.append('B{:d}_FC_{:d}_{:3.2f}'.format(branchID, i, dropout))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def AddBranchCNN1D(self, seqConvFilters, seqMaxPoolings, subsampling, seqKernelSizes, dropout, inputType, activation='relu'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_CNN1D_{:s}'.format(branchID, inputType))
    for i in range(len(seqConvFilters)):
      if i==0:
        model = Conv1D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, strides=subsampling, padding='same', name=self.GetCompatibleName('B{:d}_CNN1D_{:d}_Kernels{}_Stride_{:d}_activation_{:s}'.format(branchID, i, seqKernelSizes, subsampling, activation)))(inputLayer)
      else:
        model = Conv1D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, padding='same', name=self.GetCompatibleName('B{:d}_CNN1D_{:d}_Kernels{}_activation_{:s}'.format(branchID, i, seqKernelSizes, activation)))(model)

      if seqMaxPoolings[i] > 0:
        model = MaxPooling1D(pool_size=seqMaxPoolings[i], name='B{:d}_CNN1D_{:d}_{}'.format(branchID, i, seqMaxPoolings[i]))(model)
      if dropout:
        model = Dropout(dropout, name='B{:d}_CNN1D_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)
    model = Flatten(name='B{:d}_CNN1D_Output'.format(branchID))(model)

    self.fModelBranchesOutput.append('B{:d}_CNN1D_Output'.format(branchID))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def AddBranchLC1D(self, seqConvFilters, seqMaxPoolings, subsampling, seqKernelSizes, dropout, inputType, activation='relu'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_LC1D_{:s}'.format(branchID, inputType))
    for i in range(0, len(seqConvFilters)):
      if i==0:
        model = LocallyConnected1D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, strides=subsampling, padding='valid', name='B{:d}_LC1D_{:d}_Kernels{}_Stride_{:d}_activation_{:s}'.format(branchID, i, seqKernelSizes, subsampling, activation))(inputLayer)
      else:
        model = LocallyConnected1D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, padding='valid', name='B{:d}_LC1D_{:d}_Kernels{}_activation_{:s}'.format(branchID, i, seqKernelSizes, activation))(model)

      if seqMaxPoolings[i] > 0:
        model = MaxPooling1D(pool_size=seqMaxPoolings[i], name='B{:d}_LC1D_{:d}_{}'.format(branchID, i, seqMaxPoolings[i]))(model)
      if dropout:
        model = Dropout(dropout, name='B{:d}_LC1D_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)
    model = Flatten(name='B{:d}_LC1D_Output'.format(branchID))(model)

    self.fModelBranchesOutput.append('B{:d}_LC1D_Output'.format(branchID))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def AddBranchCNN2D(self, seqConvFilters, seqMaxPoolings, subsampling, seqKernelSizes, dropout, inputType, activation='relu'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_CNN2D_{:s}'.format(branchID, inputType))
    for i in range(0, len(seqConvFilters)):
      if i==0:
        model = Conv2D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, strides=subsampling, padding='same', name=self.GetCompatibleName('B{:d}_CNN2D_{:d}_Kernels{}_Stride_{:d}_activation_{:s}'.format(branchID, i, seqKernelSizes, subsampling, activation)))(inputLayer)
      else:
        model = Conv2D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, padding='same', name=self.GetCompatibleName('B{:d}_CNN2D_{:d}_Kernels{}_activation_{:s}'.format(branchID, i, seqKernelSizes, activation)))(model)

      if seqMaxPoolings[i] > 0:
        model = MaxPooling2D(pool_size=(seqMaxPoolings[i], seqMaxPoolings[i]), name='B{:d}_CNN2D_{:d}_{}'.format(branchID, i, seqMaxPoolings[i]))(model)
      if dropout:
        model = Dropout(dropout, name='B{:d}_CNN2D_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)
    model = Flatten(name='B{:d}_CNN2D_Output'.format(branchID))(model)

    self.fModelBranchesOutput.append('B{:d}_CNN2D_Output'.format(branchID))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def AddBranchLC2D(self, seqConvFilters, seqMaxPoolings, subsampling, seqKernelSizes, dropout, inputType, activation='relu'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_LC2D_{:s}'.format(branchID, inputType))
    for i in range(0, len(seqConvFilters)):
      if i==0:
        model = LocallyConnected2D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, strides=subsampling, padding='valid', name=self.GetCompatibleName('B{:d}_LC2D_{:d}_Kernels{}_Stride_{:d}_activation_{:s}'.format(branchID, i, seqKernelSizes, subsampling, activation)))(inputLayer)
      else:
        model = LocallyConnected2D(seqConvFilters[i], seqKernelSizes[i], activation=activation, kernel_initializer=self.fInit, padding='valid', name=self.GetCompatibleName('B{:d}_LC2D_{:d}_Kernels{}_activation_{:s}'.format(branchID, i, seqKernelSizes, activation)))(model)

      if seqMaxPoolings[i] > 0:
        model = MaxPooling2D(pool_size=(seqMaxPoolings[i], seqMaxPoolings[i]), name='B{:d}_LC2D_{:d}_{}'.format(branchID, i, seqMaxPoolings[i]))(model)
      if dropout:
        model = Dropout(dropout, name='B{:d}_LC2D_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)
    model = Flatten(name='B{:d}_LC2D_Output'.format(branchID))(model)

    self.fModelBranchesOutput.append('B{:d}_LC2D_Output'.format(branchID))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)


  ###############################################
  def AddBranchLSTM(self, outputDim, nLayers, dropout, inputType, activation='tanh'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_LSTM_{:s}'.format(branchID, inputType))
    for i in range(nLayers):
      if i==nLayers-1: # last layer
        model = LSTM(outputDim, activation=activation, name='B{:d}_LSTM_{:d}_{:d}_activation_{:s}'.format(branchID, i, outputDim, activation))(inputLayer if i==0 else model)
      else: # other layers
        model = LSTM(outputDim, return_sequences=True, activation=activation, name='B{:d}_LSTM_{:d}_{:d}_activation_returnSeq_{:s}'.format(branchID, i, outputDim, activation))(inputLayer if i==0 else model)
      
      if dropout:
        model = Dropout(dropout, name='B{:d}_LSTM_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)

    self.fModelBranchesOutput.append('B{:d}_LSTM_{:d}_{:3.2f}'.format(branchID, i, dropout))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def AddBranchRNN(self, outputDim, nLayers, dropout, inputType, activation='tanh'):
    self.fRequestedData.append(inputType)
    branchID = len(self.fTempModelBranches)

    inputLayer = Input(shape=self.fDataGenerator.GetDataTypeShape(inputType), name='B{:d}_LSTM_{:s}'.format(branchID, inputType))
    for i in range(nLayers):
      if i==nLayers-1: # last layer
        model = SimpleRNN(outputDim, activation=activation, name='B{:d}_LSTM_{:d}_{:d}_activation_{:s}'.format(branchID, i, outputDim, activation))(inputLayer if i==0 else model)
      else: # other layers
        model = SimpleRNN(outputDim, return_sequences=True, activation=activation, name='B{:d}_LSTM_{:d}_{:d}_activation_returnSeq_{:s}'.format(branchID, i, outputDim, activation))(inputLayer if i==0 else model)

      if dropout:
        model = Dropout(dropout, name='B{:d}_LSTM_{:d}_{:3.2f}'.format(branchID, i, dropout))(model)

    self.fModelBranchesOutput.append('B{:d}_LSTM_{:d}_{:3.2f}'.format(branchID, i, dropout))
    self.fModelBranchesInput.append(inputLayer.name)
    self.fTempModelBranchesInput.append(inputLayer)
    self.fTempModelBranches.append(model)

  ###############################################
  def SetFinalLayer(self, nLayers, nNeuronsPerLayer, dropout, activation='relu'):
    self.fFinalLayerStructure = []
    self.fFinalLayerNumLayers = nLayers
    self.fFinalLayerNumNeuronsPerLayer = nNeuronsPerLayer
    self.fFinalLayerDropout = dropout
    self.fFinalLayerActivation = activation

  ###############################################
  def SetFinalLayerVariable(self, layers, dropout, activation='relu'):
    self.fFinalLayerStructure = layers
    self.fFinalLayerNumLayers = 0
    self.fFinalLayerNumNeuronsPerLayer = 0
    self.fFinalLayerDropout = dropout
    self.fFinalLayerActivation = activation

  ###############################################
  def GetCompatibleName(self, name):
    return name.replace(', ', '_').replace('[','_').replace(']_','_').replace(']','_').replace(' ','_')

#__________________________________________________________________________________________________________
