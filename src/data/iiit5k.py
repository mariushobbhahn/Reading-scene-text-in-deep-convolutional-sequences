import os.path
import scipy.io

from data.helper import HelperData


class IIIT5KHelper(HelperData):
    """
       Helper to produce [image, label] from IIIT5K dataset,
    """

    def __init__(self, train_or_test, is_char_data):
        super(IIIT5KHelper, self).__init__("IIIT5K", train_or_test, is_char_data)

    def _load_mat_file(self):
        """
        Loads the test or train matlab file with the given name.
        :return: A dictionary representing the content of the matlab file.
        """
        name = "CharBound"
        path = os.path.join(self.data_dir(), self.train_or_test + name + ".mat")
        print(path)
        file = scipy.io.loadmat(path)
        data_key = self.train_or_test + name
        return file[data_key][0, ]

    def _load_char_bounds(self):
        """
        Loads the image paths, labels and corresponding char bounds.
        :return:
        """

        print("Load IIIT5K from " + self.data_dir())

        data = self._load_mat_file()
        out = []

        for i in range(0, len(data)):
            path = data[i][0][0]
            label = data[i][1][0]
            bounds = data[i][2]

            out.append((os.path.join(self.data_dir(), path), label.lower(), bounds))

        return out
