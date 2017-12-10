import os.path
import scipy.io
from data.dataset import DataSet


__all__ = ['IIIT5K']


class IIIT5K(DataSet):
    """
       Produces [image, label] in IIIT5K dataset,
    """

    DATASET_NAME = "IIIT5K"

    def load_data(self):
        """Loads the matching image paths and labels."""

        print("Load IIIT5K from " + self.dir)

        mat_file = os.path.join(self.dir, self.train_or_test + "data.mat")

        # load the data set form the mat file
        data_key = self.train_or_test + "data"

        # Take the array out of the dict and strip the first dimension
        data = scipy.io.loadmat(mat_file)[data_key][0, ]

        files = []
        labels = []

        for i in range(0, len(data)):
            files.append(os.path.join(self.dir, data[i][0][0]))
            labels.append(data[i][1][0])

        print("Loaded {} image paths and {} labels".format(len(files), len(labels)))
        return files, labels
