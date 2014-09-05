birdid_utils.py
==================

Utility routines for content based image classification using the bag of 
visual words approach, based on the phow_caltech101.py port by Ludwig Schmidt-Hackenberg, which was itself based on the phow_caltech101.m Matlab script by Andrea Vedaldi.

The routines are  a Python version of utility routines from [phow_caltech101.m][1], a 'one file' example script using the [VLFeat library][6].


The code also works with other datasets if the images are organized like in the Calltech data set, where all images belonging to one class are in the same folder:
    
    .
    |-- path_to_folders_with_images
    |    |-- class1
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    |-- some_image1.jpg
    |    |    └ ...
    |    |-- class2
    |    |    └ ...
    |    |-- class3
        ...
    |    └-- classN

There are no constraints for the names of the files or folders. File extensions can be configured in [`conf.extensions`][7] But note that the code fails with a segmentation fault (on Mac OS X 10.8.5, at least) when the images are PNGs.

These routines are exactly the same as the code in phow_caltech101.py by Ludwig Schmidt-Hackenberg, they have merely been repackaged as a module to make them 
easier to use in other contexts, i.e. in code that splits out training the 
model (which only need be done once) from the classification task it was
trained for (which may be run many times).

Requisite:

- [VLFeat with a Python wrapper][2]
- [scikit-learn][5] to replace VLFeat ML functions that don't have a Python wrapper yet. 
- [The Caltech101 dataset][3]

[5]: http://scikit-learn.org/stable/
[4]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
[2]: https://pypi.python.org/pypi/pyvlfeat/
[3]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
[1]: http://www.vlfeat.org/applications/caltech-101-code.html
[6]: http://www.vlfeat.org/index.html
[7]: https://github.com/shackenberg/phow_caltech101.py/blob/master/phow_caltech101.py#L58
