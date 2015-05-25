#!/usr/bin/env python

"""
Python rewrite of http: //www.vlfeat.org/applications/caltech-101-code.html.
Modularized for use in BirdID project by lbarnett@richmond.edu
"""
from os.path import exists, isdir, basename, join, splitext
from os import makedirs
from glob import glob
from random import sample, seed
from scipy import ones, mod, arange, array, where, ndarray, hstack, linspace, histogram, vstack, amax, amin
from scipy.misc import imread, imresize
from scipy.cluster.vq import vq
import numpy
from vl_phow import vl_phow
from vlfeat import vl_ikmeans
import pylab as pl
from datetime import datetime


SAVETODISC = False
FEATUREMAP = True
TINYPROBLEM = False
VERBOSE = True  # set to 'SVM' if you want to get the svm output
MULTIPROCESSING = False


class Configuration(object):
    def __init__(self, identifier='', prefix=''):
        self.identifier = identifier
        self.prefix = prefix
        self.inputDir = '../../../data/2014/res_med/vlfeat_training_jpg'

        # Path where training data will be stored
        self.dataDir = '../tempresults'  # should be resultDir or so
        if not exists(self.dataDir):
            makedirs(self.dataDir)
            print "folder " + self.dataDir + " created"
        self.autoDownloadData = False

        # Sum of these two numbers should be <= # of images in smallest
        # class
        self.numTrain = 30
        self.numTest = 15
        
        self.imagesperclass = self.numTrain + self.numTest
        self.numClasses = 9
        self.numWords = 600
        self.numSpatialX = [2, 4]
        self.numSpatialY = [2, 4]
        self.quantizer = 'vq'  # kdtree from the .m version not implemented
        self.svm = SVMParameters(C=10)

        # These dsift sizes are the best for the all species tests 
        self.phowOpts = PHOWOptions(Verbose=False, Sizes=[2,4,6,8],
            Step=3)
        self.clobber = False
        self.tinyProblem = TINYPROBLEM
        self.prefix = prefix
        self.randSeed = 11
        self.verbose = True
        self.extensions = [".jpg", ".jpeg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
        self.images_for_histogram = 30
        self.numbers_of_features_for_histogram = 100000

        generate_result_paths(self)
        
        if self.tinyProblem:
            print "Using 'tiny' protocol with different parameters than the .m code"
            self.prefix = 'tiny'
            self.numClasses = 5
            self.images_for_histogram = 10
            self.numbers_of_features_for_histogram = 1000
            self.numTrain
            self.numSpatialX = 2
            self.numWords = 100
            self.numTrain = 2
            self.numTest = 2
            self.phowOpts = PHOWOptions(Verbose=2, Sizes=7, Step=5)

        # tests and conversions
        self.phowOpts.Sizes = ensure_type_array(self.phowOpts.Sizes)
        self.numSpatialX = ensure_type_array(self.numSpatialX)
        self.numSpatialY = ensure_type_array(self.numSpatialY)
        if (self.numSpatialX != self.numSpatialY).any():
            messageformat = [str(self.numSpatialX), str(self.numSpatialY)]
            message = "(self.numSpatialX != self.numSpatialY), because {0} != {1}".format(*messageformat)
            raise ValueError(message)
				
	def setClasses(self, classes):
		self.classes = classes

def generate_result_paths(conf):
    conf.vocabPath = join(conf.dataDir, conf.identifier + '-vocab.py.mat')
    conf.histPath = join(conf.dataDir, conf.identifier + '-hists.py.mat')
    conf.modelPath = join(conf.dataDir, conf.prefix + '-' + conf.identifier + '-model.py.mat')
    conf.resultPath = join(conf.dataDir, conf.prefix + '-' + conf.identifier + '-result')
        

def ensure_type_array(data):
    if (type(data) is not ndarray):
        if (type(data) is list):
            data = array(data)
        else:
            data = array([data])
    return data


def standardizeImage(im):
    im = array(im, 'float32') 
    if im.shape[0] > 480:
        resize_factor = 480.0 / im.shape[0]  # don't remove trailing .0 to avoid integer devision
        im = imresize(im, resize_factor)
    if amax(im) > 1.1:
        im = im / 255.0
    assert((amax(im) > 0.01) & (amax(im) <= 1))
    assert((amin(im) >= 0.00))
    return im


def getPhowFeatures(imagedata, phowOpts):
    im = standardizeImage(imagedata)
    frames, descrs = vl_phow(im,
                             verbose=phowOpts.Verbose,
                             sizes=phowOpts.Sizes,
                             step=phowOpts.Step)
    return frames, descrs


def getImageDescriptor(model, im, conf):
    im = standardizeImage(im)
    height, width = im.shape[:2]
    numWords = model.vocab.shape[1]

    frames, descrs = getPhowFeatures(im, conf.phowOpts)
    # quantize appearance
    if model.quantizer == 'vq':
        binsa, _ = vq(descrs.T, model.vocab.T)
    elif model.quantizer == 'kdtree':
        raise ValueError('quantizer kdtree not implemented')
    else:
        raise ValueError('quantizer {0} not known or understood'.format(model.quantizer))

    hist = []
    for n_spatial_bins_x, n_spatial_bins_y in zip(model.numSpatialX, model.numSpatialX):
        binsx, distsx = vq(frames[0, :], linspace(0, width, n_spatial_bins_x))
        binsy, distsy = vq(frames[1, :], linspace(0, height, n_spatial_bins_y))
        # binsx and binsy list to what spatial bin each feature point belongs to
        if (numpy.any(distsx < 0)) | (numpy.any(distsx > (width/n_spatial_bins_x+0.5))):
            print 'something went wrong'
            import pdb; pdb.set_trace()
        if (numpy.any(distsy < 0)) | (numpy.any(distsy > (height/n_spatial_bins_y+0.5))):
            print 'something went wrong'
            import pdb; pdb.set_trace()

        # combined quantization
        number_of_bins = n_spatial_bins_x * n_spatial_bins_y * numWords
        temp = arange(number_of_bins)
        # update using this: http://stackoverflow.com/questions/15230179/how-to-get-the-linear-index-for-a-numpy-array-sub2ind
        temp = temp.reshape([n_spatial_bins_x, n_spatial_bins_y, numWords])
        bin_comb = temp[binsx, binsy, binsa]
        hist_temp, _ = histogram(bin_comb, bins=range(number_of_bins+1), density=True)
        hist.append(hist_temp)

    hist = hstack(hist)
    hist = array(hist, 'float32') / sum(hist)
    return hist


class Model(object):
    def __init__(self, classes, conf, vocab=None):
        self.classes = classes
        self.phowOpts = conf.phowOpts
        self.numSpatialX = conf.numSpatialX
        self.numSpatialY = conf.numSpatialY
        self.quantizer = conf.quantizer
        self.vocab = vocab


class SVMParameters(object):
    def __init__(self, C):
        self.C = C


class PHOWOptions(object):
    def __init__(self, Verbose, Sizes, Step):
        self.Verbose = Verbose
        self.Sizes = Sizes
        self.Step = Step


def get_classes(datasetpath, numClasses):
    classes_paths = [files
                     for files in glob(datasetpath + "/*")
                     if isdir(files)]
    classes_paths.sort()
    classes = [basename(class_path) for class_path in classes_paths]
    if len(classes) == 0:
       raise ValueError('no classes found')
    if len(classes) < numClasses:
       raise ValueError('conf.numClasses is bigger than the number of folders')
    classes = classes[:numClasses]
    return classes


def get_imgfiles(path, extensions):
    all_files = []
    all_files.extend([join(path, basename(fname))
                     for fname in glob(path + "/*")
                     if splitext(fname)[-1].lower() in extensions])
    return all_files


def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


def get_all_images(classes, conf):
    all_images = []
    all_images_class_labels = []
    for i, imageclass in enumerate(classes):
        path = join(conf.inputDir, imageclass)
        extensions = conf.extensions
        imgs = get_imgfiles(path, extensions)
        if len(imgs) == 0:
            raise ValueError('no images for class ' + str(imageclass))

        if conf.numTrain > 0:
            imgs = sample(imgs, conf.imagesperclass)
        all_images = all_images + imgs

        if conf.numTrain > 0:
            class_labels = list(i * ones(conf.imagesperclass))
        else:
            class_labels = list(i * ones(len(imgs)))

        all_images_class_labels = all_images_class_labels + class_labels

    all_images_class_labels = array(all_images_class_labels, 'int')
    return all_images, all_images_class_labels


def create_split(all_images, conf):
    temp = mod(arange(len(all_images)), conf.imagesperclass) < conf.numTrain
    selTrain = where(temp == True)[0]
    selTest = where(temp == False)[0]
    # the '[0]' is there, because 'where' returns tuples, don't know why....
    # the use of the 'temp' variable is not pythonic, but we need the indices 
    # not a boolean array. See Matlab code
    return selTrain, selTest

def create_split_n(all_images, imgsPerClass, numTrain):
    temp = mod(arange(len(all_images)), imgsPerClass) < numTrain
    selTrain = where(temp == True)[0]
    selTest = where(temp == False)[0]

    return selTrain, selTest


def trainVocab(selTrain, all_images, conf):
    selTrainFeats = sample(selTrain, conf.images_for_histogram)
    descrs = []
    if MULTIPROCESSING:
        raise ValueError('MULTIPROCESSING not implemented')
        #pool = Pool(processes=30)  
        #list_of_train_images = [all_images[i] for i in selTrainFeats]
        #descrs.append(pool.map_async(getPhowFeatures, list_of_train_images).get())        
    else:
        for i in selTrainFeats:
            im = imread(all_images[i])
            # Debugging
            #print all_images[i], 'shape', im.shape
            descrs.append(getPhowFeatures(im, conf.phowOpts)[1])
            # the '[1]' is there because we only want the descriptors and not the frames
    
    descrs = hstack(descrs)
    n_features = descrs.shape[1]
    sample_indices = sample(arange(n_features), conf.numbers_of_features_for_histogram)
    descrs = descrs[:, sample_indices]
    descrs = array(descrs, 'uint8')
    
    # Quantize the descriptors to get the visual words
    vocab, _ = vl_ikmeans(descrs,
                          K=conf.numWords,
                          verbose=conf.verbose,
                          method='elkan')
    return vocab


def computeHistograms(all_images, model, conf):
    hists = []
    for ii, imagefname in enumerate(all_images):
        #print('Processing {0} ({1:.2f}%)'.format(imagefname, 100.0 * ii / len(all_images)))
        im = imread(imagefname)
        hists_temp = getImageDescriptor(model, im, conf)
        hists.append(hists_temp)
    hists = vstack(hists)
    return hists

