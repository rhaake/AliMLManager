from __future__ import print_function
import ROOT
import multiprocessing, numpy, copy, logging, os.path


#__________________________________________________________________________________________________________
class AliMLDataLoader:
  """Class to create data samples from raw data input
  The data is extracted sequentially
  Usage:
  * Initialize the loader with a data generator
  * Add at least one class with its cuts using AddClass()
  * Get a data array with GetDataChunk (e.g. for val data)
  * Get another data array for training data
  * Optional instead:
    * Start threaded loading of data with StartLoading
    * Get the data from the queue
  """

  ###############################################
  def __init__ (self, dataGenerator, requested_data, numSamplesPerChunk):
    # Internal thread/queue that holds ready data arrays
    self.fThread = None
    self.fQueue  = multiprocessing.Queue(maxsize=1)
    self.fDataGenerator = dataGenerator

    # List of used classes (only one is necessary for regression tasks)
    self.fClasses = []
    self.fRequestedData = requested_data

    self.fProducedSamples    = 0
    self.fMaxProducedSamples = 10000000
    self.fNumSamplesPerChunk = numSamplesPerChunk
    self.fRawDataIndices     = []
    self.fCurrentNumSamples  = 0
  ###############################################
  def ResetCounters(self):
    """Reset counters"""
    for myClass in self.fClasses:
      for index in myClass['indices']:
        index = 0

  ###############################################
  def StartLoading(self):
    """Start filling the queue"""
    # Create a process generating the training data by filling it into the queue
    self.fProducedSamples = 0
    self.fThread = multiprocessing.Process(target=self._LoaderProcess, args=(self.fQueue,))
    self.fThread.start()

  ###############################################
  def AddClass(self, datasets_in, cuts):
    """Add class used to fill the dataset"""
    # Input e.g.:
    #  datasets = [{'file': './a.root', 'treename': 'ExtractedJets', 'weight': 0.8 }, ]
    #  cuts = [{'branch': 'Jets', 'var': 'TruePt', 'min': 10., 'max': 100.}, {'branch': 'Jets', 'var': 'PID', 'isint': [0,3]}]

    datasets = copy.deepcopy(datasets_in)

    # In case we only have 1 dataset, automatically set the weight to one
    if len(datasets) == 1:
      if 'weight' in datasets[0].keys() and datasets[0]['weight'] != 1:
        logging.info('Dataset weight changed to 1, since it is the only set in the class (dataset: {:s})'.format(datasets[0]['treename']))
      datasets[0]['weight'] = 1

    # Check if dataset weight settings are OK
    totalWeight = 0
    for dset in datasets:
      totalWeight += dset['weight']
    if round(totalWeight, 3) !=  1.000:
      raise ValueError('The weights for the class do not sum up to 1, but to {}'.format(totalWeight))

    # Set properties of the class (trees will be loaded later)
    logging.info('Adding data class {:d}.'.format(len(self.fClasses)))
    newClass = {}
    newClass['datasets'] = datasets
    newClass['indices']   = [0, ] * len(datasets)
    newClass['cuts']  = cuts
    newClass['id']    = len(self.fClasses)

    self.fClasses.append(newClass)

  ###############################################
  def FastForward(self, nSamples):
    """Fast forward some events"""

    if nSamples == 0:
      return

    totalSkippedSamples = 0
    for myClass in self.fClasses:
      for iData in range(len(myClass['datasets'])):
        dataset = myClass['datasets'][iData]
        fileH = ROOT.TFile.Open(dataset['file'])
        tree = fileH.Get(dataset['treename'])
        numSamplesPerDataset = round(nSamples*dataset['weight'], 0)

        skippedSamples = 0
        while skippedSamples < numSamplesPerDataset:
          currentRawData = self._GetNextRawSample(myClass, tree, iData)
          if not currentRawData:
            break
          skippedSamples += 1
          totalSkippedSamples += 1
        fileH.Close()

    if totalSkippedSamples < nSamples:
      logging.warning('Warning: Fast forwarded to the end of the dataset!')


  ###############################################
  def GetNumSamples(self):
    """Calculate how many samples are all datasets"""

    # Reset counters to loop over full dataset
    oldIndices = []
    for myClass in self.fClasses:
      for index in myClass['indices']:
        oldIndices.append(index)
    self.ResetCounters()

    totalNumSamples = 0
    for myClass in self.fClasses:
      for iData in range(len(myClass['datasets'])):
        dataset = myClass['datasets'][iData]
        fileH = ROOT.TFile.Open(dataset['file'])
        tree = fileH.Get(dataset['treename'])
        while True:
          currentRawData = self._GetNextRawSample(myClass, tree, iData)
          if not currentRawData:
            break
          totalNumSamples += 1
        fileH.Close()

    # Restore old counter state
    i = 0
    for myClass in self.fClasses:
      for index in range(len(myClass['indices'])):
        myClass['indices'][index] = oldIndices[i]
        i += 1

    return totalNumSamples

  ###############################################
  def GetDataChunk(self, numSamplesPerClass):
    """Returns a data array for a list of classes in the form (arr_data, arr_truth) """

    #### First calculate number of total samples
    totalNumSamples = 0
    for myClass in self.fClasses:
      for dset in myClass['datasets']:
        totalNumSamples += int(round(numSamplesPerClass * dset['weight'], 0))

    offset = 0
    data = []
    #### Create empty arrays
    truth  = numpy.zeros(shape=(totalNumSamples))
    for datatype in self.fRequestedData:
      arr = self.fDataGenerator.CreateEmptyArray(datatype, totalNumSamples)
      data += [arr]

    # Prepare the list containing indices for the used data
    self.fRawDataIndices = []
    self.fCurrentNumSamples = 0

    #### Fill the dataset, class-per-class
    for iClass in range(len(self.fClasses)):
      # Use the correct offset when filling the array
      offset += self._FillDataChunk(data, truth, offset, iClass, numSamplesPerClass)

    # If the dataset has not been filled fully (not enough data left) return
    if offset != totalNumSamples:
      logging.warning('Not enough events to fill the data array (got {:d}/{:d}).'.format(offset, totalNumSamples))
      return None

    self.fProducedSamples += offset

    return (data, truth, numSamplesPerClass, self.fRawDataIndices)


  ###############################################
  def _LoaderProcess(self, queue):
    """Run loop to get data in chunks"""
    while self.fProducedSamples < self.fMaxProducedSamples:
      array = self.GetDataChunk(self.fNumSamplesPerChunk)
      if array == None:
        queue.put(None)
        break
      queue.put(array)

  ###############################################
  def _FillDataChunk(self, data, truth, offset, iClass, numSamplesPerClass):
    """Fill data in the empty arrays for class myClass"""

    myClass = self.fClasses[iClass]
    totalSamples = 0

    #### Loop over all datasets in this class
    for iData in range(len(myClass['datasets'])):
      # Open input root tree
      dataset = myClass['datasets'][iData]
      fileH = ROOT.TFile.Open(dataset['file'])
      tree = fileH.Get(dataset['treename'])
      numSamplesPerDataset = int(round(numSamplesPerClass*dataset['weight'], 0))

      samplesInDataset = 0
      # Loop that fills numSamplesPerDataset into array OR breaks earlier if not available
      while samplesInDataset < numSamplesPerDataset:
        # Get next sample in dataset (that passed cuts)

        currentRawData = self._GetNextRawSample(myClass, tree, iData)
        if not currentRawData:
          break

        # Save raw data indices/trees for analysis
        # This is useful if you want to do something directly with the raw data
        self.fRawDataIndices.append(self.fCurrentNumSamples)

        # Use the raw data to create the demanded sample
        currentTruth = None
        for iType in range(len(self.fRequestedData)):
          datatype = self.fRequestedData[iType]
          # ... and load them using the generator
          currentSample = self.fDataGenerator.GetSample(currentRawData, datatype, myClass['indices'][iData]-1)
          data[iType][offset+totalSamples]  = currentSample

          # Load the truth once
          if not currentTruth:
            currentTruth  = self.fDataGenerator.GetSampleTruth(currentRawData, myClass['id'])
            truth[offset+totalSamples] = currentTruth

        self.fCurrentNumSamples += 1
        samplesInDataset += 1
        totalSamples += 1

      fileH.Close()
      logging.info('Class {:d}: Created {:6d} events with dataset={:s}'.format(myClass['id'], samplesInDataset, dataset['treename']))

    return totalSamples


  ###############################################
  def _GetNextRawSample(self, myClass, tree, dsetID):
    """Get the next sample in line that passes the cuts"""
    rawSample = None
    # Search for the next sample that fulfills the cuts
    while True:
      if myClass['indices'][dsetID] >= tree.GetEntriesFast():
        break

      # The raw sample is the filled tree handle
      tree.GetEntry(myClass['indices'][dsetID])
      myClass['indices'][dsetID] += 1
      if not self._IsCutFulfilled(tree, myClass['cuts']):
        continue
      else:
        rawSample = tree
        break

    return rawSample


  ###############################################
  def _IsCutFulfilled(self, currentRawData, cuts):
    """Check if currentRawData passed the cuts"""
    # TODO: Better apply the cuts using a ROOT.TCut object

    passed = True
    for cut in cuts:
      sample = getattr(currentRawData, cut['branch']) # each cut must contain the branch it applies to
      cutVar = getattr(sample, cut['var'])
      if callable(cutVar):
        cutVar = cutVar()
      # Apply min/max cut
      if 'min' in cut.keys() and 'max' in cut.keys():
        if cutVar >= cut['max'] or cutVar < cut['min']:
          passed = False
      elif 'isint' in cut.keys():
        if isinstance(cut['isint'], list):
          if int(cutVar) not in cut['isint']:
            passed = False
        else:
          if int(cutVar) != cut['isint']:
            passed = False
      else:
        raise ValueError('Cut {} not recognized.'.format(cut))


    return passed


  ###############################################
  def CreateTreeFromRawData(self, fname, tname, scores=None, filterList=None, rawDataIndices=None):
    """Raw data selected by data creation procedure
       The purpose of this function is to get a tree of raw data
       in parallel to the data samples for the learning task
    """
    if rawDataIndices != None:
      self.fRawDataIndices = rawDataIndices

    if self.fRawDataIndices == None:
      raise ValueError('Data for CreateTreeFromRawData() must be created by GetDataChunk() before or explicitly given')

    ##### Define chain of input files
    chain = ROOT.TChain()
    for iClass, myClass in enumerate(self.fClasses):
      for iData in range(len(myClass['datasets'])):
        # Load dataset
        dataset = myClass['datasets'][iData]
        chain.Add('{:s}/{:s}'.format(dataset['file'], dataset['treename']))

    ##### Define output
    ofile   = ROOT.TFile(fname, 'update')
    outTree = chain.CloneTree(0)
    outTree.SetName(tname)
    score   = numpy.zeros(1, dtype=float)
    brScore = outTree.Branch('MLScores', score, 'Scores/D')


    ##### Loop through the chain and add raw samples to output tree
    for iSample, sampleIndex in enumerate(self.fRawDataIndices):
      if filterList != None and iSample not in filterList:
        continue
      chain.GetEntry(sampleIndex)
      score[0] = scores[iSample]
      outTree.Fill()

    outTree.AutoSave()
    ofile.Close()

  ###############################################
  def GetRawDataChain(self):
    """Get raw data chain (no cuts applied)"""

    ##### Define chain of input files
    chain = ROOT.TChain()
    for iClass, myClass in enumerate(self.fClasses):
      for iData in range(len(myClass['datasets'])):
        # Load dataset
        dataset = myClass['datasets'][iData]
        chain.Add('{:s}/{:s}'.format(dataset['file'], dataset['treename']))

    return chain
