import os
import numpy as np
from PIL import Image
from skimage.util import img_as_float
from skimage.color import rgb2grey
from skimage.io import imread
from scipy.io import loadmat


class CityscapesDataset(object):
    """
    Cityscapes dataset wrapper

    Given the path to the root of the Cityscapes dataset, this class provides
    methods for loading images, ground truths and evaluating predictions

    Attribtes:

    city_path - the root path of the dataset
    data_path - the path of the data directory within the root
    images_path - the path of the images directory within the data dir
    gt_path - the path of the groundTruth directory within the data dir
    train_sample_names - a list of names of training images
    val_sample_names - a list of names of validation images
    test_sample_names - a list of names of test images
    """
    def __init__(self, city_path):
        """
        Constructor

        :param city_path: the path to the root of the Cityscapes dataset
        """
        self.data_path = city_path
        self.gt_path = os.path.join(self.data_path, 'gtFine')

        self.train_sample_names = self._sample_names(self.data_path, 'train')
        self.val_sample_names = self._sample_names(self.data_path, 'val')
        self.test_sample_names = self._sample_names(self.data_path, 'test')

    @staticmethod
    def _sample_names(base, subset):
        names = []
        folders = os.listdir(os.path.join(base, subset))
        for folder in folders:
            files = os.listdir(os.path.join(base, subset, folder))
            for fn in files:
                dir, filename = os.path.split(fn)
                name, ext = os.path.splitext(filename)
                if ext.lower() == '.png':
                    names.append(os.path.join(subset, folder, name[:-11]))
        return names

    def read_image(self, name):
        """
        Load the image identified by the sample name (you can get the names
        from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a (H,W,3) array containing the image, scaled to range [0,1]
        """
        path = os.path.join(self.data_path, name + 'leftImg8bit.png')
        return img_as_float(imread(path))

    def get_image_shape(self, name):
        """
        Get the shape of the image identified by the sample name (you can
        get the names from the `train_sample_names`, `val_sample_names` and
        `test_sample_names` attributes)
        :param name: the sample name
        :return: a tuple of the form `(height, width, channels)`
        """
        path = os.path.join(self.data_path, name + 'leftImg8bit.png')
        img = Image.open(path)
        return img.height, img.width, 3

    def segmentations(self, name):
        """
        Load the ground truth segmentations identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        path = os.path.join(self.gt_path, name + 'gtFine_instanceIds.png')
        return self.load_segmentations(path)

    def boundaries(self, name):
        """
        Load the ground truth boundaries identified by the sample name
        (you can get the names from the `train_sample_names`,
        `val_sample_names` and `test_sample_names` attributes)
        :param name: the sample name
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        path = os.path.join(self.gt_path, name + 'gtFine_edgemap.png')
        return self.load_boundaries(path)

    @staticmethod
    def load_segmentations(path):
        """
        Load the ground truth segmentations from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        segmentation ground truth
        """
        gt = Image.open(path).convert('L')
        return [np.array(gt, dtype=np.int32)]

    @staticmethod
    def load_boundaries(path):
        """
        Load the ground truth boundaries from the Matlab file
        at the specified path.
        :param path: path
        :return: a list of (H,W) arrays, each of which contains a
        boundary ground truth
        """
        gt = Image.open(path).convert('L')
        w, h = gt.size
        gt = gt.resize((w // 2, h // 2), resample=Image.NEAREST)
        gt_bin = np.array(gt) / 255
        return [np.array(gt_bin, dtype=np.uint8)]