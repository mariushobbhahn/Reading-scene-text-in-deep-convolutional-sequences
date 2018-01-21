import os
import sys
import argparse

import config
import data.utils
from data.iiit5k import IIIT5KChar

#from cnn import network


def main(argv):
    print("start network: server={}".format(config.IS_SERVER))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--dump-dir', dest="dump_dir", help='dumps the used train data into the given directory.')
    parser.add_argument('--remove-lmdb', dest="remove_lmdb",help='if set, the old .mdb files will be removed.')

    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config.REMOVE_LMDB = args.remove_lmdb is not None
    config.DUMP_DIR = args.dump_dir

    df = IIIT5KChar("train", os.path.join(config.DATA_DIR, "IIIT5K"))
    df = data.utils.load_lmdb(df)

    df.reset_state()
    data.utils.dump_data(df, os.path.join(config.DATA_DIR, "dump"))


    #df.reset_state()

    #for (img, label) in df.get_data():
    #    print("Found image of size {} for label {}".format(img.shape, label))
    #config.DUMP_DATABASES = False#"-d" in argv
    #config.REMOVE_LMDB = False#"-r" in argv


    #network.run()


if __name__ == "__main__":
    main(sys.argv[1:])
