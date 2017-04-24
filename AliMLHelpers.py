from __future__ import print_function
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plot
import numpy, math, os, sys, cPickle, logging
import ROOT

logging.basicConfig(format='%(asctime)s %(message)s', stream=sys.stdout, level=logging.INFO)

ROOT.gROOT.SetBatch(True)  # don't show canvases
numpy.set_printoptions(threshold=numpy.inf)

for directory in ['./Models', './Results']:
  if not os.path.exists(directory):
    os.makedirs(directory)

from keras.layers.normalization import Layer

#__________________________________________________________________________________________________________

def NotifyError():
  #os.system('play --norm=-15 --no-show-progress -v 1 --null --channels 1 synth %s sine %f' % ( 0.06, 400))
  #os.system('play --norm=-15 --no-show-progress -v 1 --null --channels 1 synth %s sine %f' % ( 0.2, 400))
  pass

def NotifyDone():
  #os.system('play --norm=-15 --no-show-progress -v 1 --null --channels 1 synth %s sine %f' % ( 0.06, 550))
  #os.system('play --norm=-15 --no-show-progress -v 1 --null --channels 1 synth %s sine %f' % ( 0.06, 750))
  #os.system('play --norm=-15 --no-show-progress -v 1 --null --channels 1 synth %s sine %f' % ( 0.06, 550))
  pass

def SaveToRootFile(fname, obj):
  fileH = ROOT.TFile.Open(fname, 'UPDATE')
  obj.Write(obj.GetName(), ROOT.TObject.kOverwrite)
  fileH.Close()

def SavePlot(fname, title, y, functionlabels, x=[], rangex=(), rangey=(),legendloc='upper right', axislabels=(), logY=False):
  """Simple plot helper function"""
  # Check input data
  for item in x:
    if math.isnan(item):
      print('Plot {:s} was not saved due to invalid values.'.format(fname))
      return
  for i in range(len(y)):
    for item in y[i]:
      if math.isnan(item):
        print('Plot {:s} was not saved due to invalid values.'.format(fname))
        return
  if not len(x):
    x = range(len(y[0]))
  for i in range(len(y)):
    plot.plot(x, y[i], label=functionlabels[i])
  plot.legend(loc=legendloc)
  if len(rangex) == 2:
    plot.xlim(rangex[0], rangex[1])
  if len(rangey) == 2:
    plot.ylim(rangey[0], rangey[1])
  plot.title(title)
  if len(axislabels) == 2:
    plot.xlabel(axislabels[0])
    plot.ylabel(axislabels[1])

#  newfname = fname.replace('.png', '.0.png')
#  i = 1
#  while os.path.isfile(newfname):
#    newfname = fname.replace('.png', '.{:d}.png'.format(i))
#    i += 1
#  fname = newfname

  if logY:
    plot.yscale('log', nonposy='clip')

  plot.savefig("{:s}".format(fname), dpi=72)
  #plot.show()
  plot.clf()


def SaveHistogram(fname, title, y, functionlabels, rangex=(0,1), legendloc='upper right', logY=False, nbins=100):
  """Simple histogram helper function"""
  # Check input data
  for i in range(len(y)):
    for item in y[i]:
      if math.isnan(item):
        print('Histogram {:s} was not saved due to invalid values.'.format(fname))
        return


  for i in range(len(y)):
    plot.hist(y[i], label=functionlabels[i], bins=nbins, histtype='step')
  plot.legend(loc=legendloc)
  plot.title(title)
  plot.xlim(rangex[0], rangex[1])
#  plot.yscale('log', nonposy='clip')

  if logY:
    plot.yscale('log', nonposy='clip')


#  newfname = fname.replace('.png', '.0.png')
#  i = 1
#  while os.path.isfile(newfname):
#    newfname = fname.replace('.png', '.{:d}.png'.format(i))
#    i += 1
#  fname = newfname

  plot.savefig("{:s}".format(fname), dpi=72)
  #plot.show()
  plot.clf()

def CompareTwoDatasets(self, dataset1, dataset2):
  for i in range(len(dataset1)):
    for j in range(len(dataset2)):
      if (dataset1[i] == dataset2[j]).all():
        print('Data in {:d} equal to {:d}'.format(i,j))

