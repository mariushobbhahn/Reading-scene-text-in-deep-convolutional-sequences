import os.path
import scipy.io as sp

from data.utils import *
from data.dataset import NamedDataFlow


class IIIT5K(NamedDataFlow):
    """
      Base class for a DataFlow from the IIIT5K data set.
      Will return (image, label) data points, where image will be a grayscale image with height of 32px.
    """

    _cached_file = None

    def __init__(self, train_or_test, data_dir=None, unique=False, name="IIIT5K"):
        if data_dir is None:
            data_dir = os.path.join(config.DATA_DIR, name)

        self.train_or_test = train_or_test
        self.data_dir = data_dir
        self.unique = unique
        super(IIIT5K, self).__init__(name)

    def size(self):
        if (self.unique):
            return 36
        return sum(1 for _ in self._get_mat_file())

    def get_name(self):
        return self._name + "_" + self.train_or_test

    def _get_mat_file(self):
        """
          Loads the mat file containing the labels of the data set.
          Will yield every entry.
          :return: A tupel with (path, label, char_bounds)
        """

        # load file if needed
        if self._cached_file is None:

            # Key for the dictinoary
            key = self.train_or_test + "CharBound"
            path = os.path.join(self.data_dir, self.train_or_test + "CharBound.mat")

            # Tries to load the matlab file
            self._cached_file = sp.loadmat(path)[key][0, ]

        return self._cached_file

    def _get_paths(self):
        """
          Parses the image path, label and the char bounds for every data point from the mat file.
        :return: A tuple with (path, label, char_bounds).
        """
        # yield every entry of the file
        for (path, label, bounds) in self._get_mat_file():

            yield (path[0, ], label[0, ], bounds)

    def _get_images(self):
        """
        Loads the images, labels and the char bounds for every data point from the mat file.

        :return: A tuple with (image, label, char_bounds).
        """

        # yield every entry of the file
        for (path, label, bounds) in self._get_paths():

            # Load image as grayscale
            img_file = os.path.join(self.data_dir, path)
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

            yield (img, label, bounds)

    def get_data(self):
        for (img, label, _) in self._get_images():

            # Resize image if needed
            # Check if height is 32px
            h = img.shape[0]

            if h != 32:
                # Resize to 32px
                f = 32.0 / h
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            # Load image
            yield (img, label)


class IIIT5KChar(IIIT5K):
    """
      DataFlow which contains the characters from the IIIT5K data set.
      Returns (image, label) data points, where label is an integer between 0 and 35 and image a 32x32 grayscale image.
    """

    def __init__(self, train_or_test, data_dir=None, unique=False, name="IIIT5K"):
        super(IIIT5KChar, self).__init__(train_or_test=train_or_test, data_dir=data_dir, unique=unique, name=name)

    def get_name(self):
        return self._name + "_char_" + self.train_or_test

    def get_data(self):
        known_labels = set()
        last_index = -10
        index = 0
        for (img, label, char_bounds) in self._get_images():
            # convert string to list of chars
            chars = list(label)
            (img_height, img_width) = img.shape

            # print("Check size: {}x{}".format(img_width, img_height))
            # Skip images where no quadratic frame could be cut off
            if img_height > img_width:
                continue

            half_height = img_height / 2
            max_x = img_width - img_height
            scale = 32.0 / img_height

            for (char, bounds) in zip(chars, char_bounds):
                label = char_to_int_label(char)
                # Bounds is array with [x, y, w, h]
                # Cutoff quadratic images with full height, centered around the char.
                index = index + 1
                if self.unique and (label in known_labels or (index < last_index + 8)):
                    # print('no')
                    continue

                known_labels.add(label)
                last_index = index

                center_x = bounds[0] + bounds[2] / 2

                # calculated optimal x
                x = int(center_x - half_height)

                # clamp to keep inside image (0 <= x <= MAX_x)
                x = max(0, min(x, max_x))

                # print("cut image in rect ({}, {}, {}, {})".format(x, 0, img_height, img_height))

                # cut off character image
                char_img = img[0:img_height, x:(x + img_height)]

                # print("char image size: {}".format(char_img.shape))
                # Scale to 32x32
                if img_height != 32:
                    char_img = cv2.resize(char_img, None, fx=scale, fy=scale)

                # print("Yield image for char {} with label {}".format(char, char_to_int_label(char)))
                yield (char_img, label)
