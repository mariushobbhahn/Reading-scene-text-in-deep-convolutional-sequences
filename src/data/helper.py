import cv2
import numpy as np
import os.path
import config

from abc import abstractmethod

from tensorpack.dataflow.base import RNGDataFlow


def _read_image(path):
    """
    Reads the image from the path. The image will be converted to grayscale and scaled to 32px height.
    :param path: The file path of the image.
    :return: a grayscale image with 32px height.
    """

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("Read image with size {}".format(img.shape))
    # Check if height is 32px
    h = img.shape[0]

    if h != 32:
        # Resize to 32px
        f = 32.0 / h
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    return img


class HelperData(RNGDataFlow):
    """
       Produces a data set of [image, label].
    """

    def __init__(self, name, train_or_test, is_char_data):
        self.name = name
        self.is_char_data = is_char_data
        self.train_or_test = train_or_test
        self.__data = None

        super(HelperData, self).__init__()

        # reset data set
        self.reset_state()

    def reset_state(self):
        self.__data = None
        super(HelperData, self).reset_state()

    def size(self):
        return len(self._lazy_data())

    def data_dir(self):
        return os.path.join(config.DATA_DIR, self.name)

    def get_data(self):
        for img, label in self._lazy_data():
            success, raw = cv2.imencode(".png", img)

            if success:
                png = np.asarray(bytearray(raw), dtype='uint8')
                yield [png, label]

    def _lazy_data(self):
        if self.__data is None:
            self.__data = self._load_data()

        return self.__data

    def _load_data(self):
        """
        Loads the images and their labels.
        :return: a array of image and label pairs.
        """
        if self.is_char_data:
            return self._load_char_data()
        else:
            return self._load_scene_data()

    def _load_scene_data(self):
        def f(x):
            """
            Reads the image from the path and remove bounding boxes.
            :param x:
            :return:
            """
            path, label, _ = x
            return _read_image(path), label

        return map(f, self._load_char_bounds())

    def _load_char_data(self):
        """
        Loads 32x32 images from the data set which contains a single character.
        :return: a array of images and label pairs.
        """

        triples = self._load_char_bounds()
        out = []

        for tri in triples:
            # Decompose triples
            path, label, bounds = tri
            # Read image and read height
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            h = int(img.shape[0])

            chars = list(label)

            # Iterate over each char and their bounding box
            for idx in range(len(chars)):

                char = chars[idx]
                rect = bounds[idx]

                # Calc mid x of char bounds.
                x = max(int(rect[0] + (rect[2] - h) / 2), 0)
                f = 32.0 / h

                # print("Char: {} bounds: {} subimage: {}:{}, {}:{}".format(char, rect, 0, h, x, (x + h)))

                # Cut out char image and scale it to 32 x 32
                char_image = img[0:h, x:(x + h)]
                char_image = cv2.resize(char_image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

                # store char and image
                out.append((char_image, char))

        return out

    @abstractmethod
    def _load_char_bounds(self):
        """
        Loads the paths to the images, the labels and the char bounds inside each image.

        :return: a array of paths, labels and array of char bounds.
        """


