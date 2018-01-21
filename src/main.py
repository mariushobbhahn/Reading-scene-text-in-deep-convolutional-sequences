import os
import sys

import config
from data.iiit5k import IIIT5KChar

#from cnn import network


def main(argv):
    df = IIIT5KChar("train", os.path.join(config.DATA_DIR, "IIIT5K"))
    df.reset_state()

    for (img, label) in df.get_data():
        print("Found image of size {} for label {}".format(img.shape, label))
    #config.DUMP_DATABASES = False#"-d" in argv
    #config.REMOVE_LMDB = False#"-r" in argv


    #network.run()


if __name__ == "__main__":
    main(sys.argv[1:])
