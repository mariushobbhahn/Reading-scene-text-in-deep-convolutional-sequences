from tensorpack.dataflow import DataFlow


class NamedDataFlow(DataFlow):
    """
      Subclass of DataFlow which
    """
    def __init__(self, name):
        self._name = name

    def get_name(self):
        """
        Returns the name of this data set. The name can depend on the train or test config.
        :return:
        """
        return self._name

