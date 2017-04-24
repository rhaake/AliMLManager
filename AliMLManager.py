from __future__ import print_function
import logging
import AliMLLoader

#######################################################################################################
def DoTraining(model, dataset, generator, numEpochs, numEventsTraining, numEventsValidation, eventChunkSize, eventOffset):
  """Get data + train on batch iteratively"""
  ###############
  # Setup jet loader instance
  dataLoader = AliMLLoader.AliMLDataLoader(generator, model.fRequestedData, eventChunkSize)
  dataLoader.FastForward(eventOffset)
  dataLoader.fMaxProducedSamples = numEventsTraining

  # add classes defined in the dataset
  for dclass in dataset:
    dataLoader.AddClass(dclass['datasets'], dclass['cuts'])

  ###############
  # Load validation data
  logging.info('Loading validation data... ')
  (mergedValidationData, mergedValidationTruth, samplesPerClass, _)  = dataLoader.GetDataChunk(numEventsValidation)

  # Separate the data class-wise
  innerList = [list([]) for _ in xrange(samplesPerClass)]
  X_test = [ list(innerList) for _ in xrange(len(mergedValidationTruth)/samplesPerClass) ]
  y_test  = []
  for i in range(0, len(mergedValidationTruth)/samplesPerClass):
    for dtype in range(len(mergedValidationData)):
      X_test[i][dtype] = (mergedValidationData[dtype][i*samplesPerClass:(i+1)*samplesPerClass])
    y_test.append(mergedValidationTruth[i*samplesPerClass:(i+1)*samplesPerClass])

  ###############
  # Training loop:
  # - Loop over chunked data (1 chunk if data fits in memory)
  # - Chunks loaded asynchronously into the queue

  if int(numEventsTraining/eventChunkSize) == 1:
    logging.info('Training of {:d} samples in one chunk...'.format(numEventsTraining*len(dataLoader.fClasses)))
    logging.info('Loading training data... ')
    array = dataLoader.GetDataChunk(numEventsTraining)
    model.TrainModel(array[0], array[1], X_test, y_test, mergedValidationData, mergedValidationTruth, numEpochs=numEpochs)
    model.SaveModel()
  else:
    chunk = 0
    dataLoader.StartLoading()
    logging.info('Training of {:d} samples in {:d} chunks...'.format(numEventsTraining*len(dataLoader.fClasses), numEventsTraining/eventChunkSize))
    while dataLoader.fThread.is_alive() or not dataLoader.fQueue.empty():
      queueResult = dataLoader.fQueue.get()
      if queueResult == None:
        break
      model.TrainModel(queueResult[0], queueResult[1], X_test, y_test, mergedValidationData, mergedValidationTruth, numEpochs=numEpochs)
      chunk += 1

      logging.info('Chunk {:d} done. {:d}/{:d} samples trained in total.\n'.format(chunk, chunk*eventChunkSize*len(dataLoader.fClasses), numEventsTraining*len(dataLoader.fClasses)))
    model.SaveModel()

  ###############
  dataLoader.ResetCounters()
