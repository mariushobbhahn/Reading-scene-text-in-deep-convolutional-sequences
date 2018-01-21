import os
import sys
import argparse
import config

from cnn import network


def main(argv):
    print("start network: server={}".format(config.IS_SERVER))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--dump-dir', dest="dump_dir", help='dumps the used train data into the given directory.')
    parser.add_argument('-r', action='store_true', dest="remove_lmdb", help='if set, the old .mdb files will be removed.')

    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("Remove {}".format(args.remove_lmdb))

    config.REMOVE_LMDB = args.remove_lmdb
    config.DUMP_DIR = args.dump_dir

    network.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
