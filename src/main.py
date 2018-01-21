import os
import sys

import config
from cnn import network


def main(argv):
    config.DUMP_DATABASES = True#"-d" in argv
    config.REMOVE_LMDB = False#"-r" in argv


    network.run()


if __name__ == "__main__":
    main(sys.argv[1:])
