import os
import sys
import argparse
import config


from cnn.train import train
from cnn.test import test


def main(argv):
    print("start network: server={}".format(config.IS_SERVER))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--dump', action='store_true', help='dumps the used train data.')

    # control the batch size
    parser.add_argument('--batch-size', dest='batch_size', help='The used batch size for training.')
    parser.add_argument('-b', dest='batch_size', help='The used batch size for training.')

    parser.add_argument('--unique', action='store_true', help='if set only one data point per label will be used for training.')
    parser.add_argument('--sub-data', dest='sub_data', help='uses only the given amount of data points.')
    parser.add_argument('-r', action='store_true', dest="remove_lmdb", help='if set, the old .mdb files will be removed.')

    #TODO Implement test mode
    parser.add_argument('--test', help='predicts the characters in the image at the given path')

    # parse arguments
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config.REMOVE_LMDB = args.remove_lmdb
    config.DUMP_DIR = os.path.join(config.RES_DIR, 'dump') if args.dump else None

    if args.test:
        model = args.load or os.path.join(config.RES_DIR, 'train_log/jan0226-223108/checkpoint')
        test(args.test, model)
    else:
        # start training
        train(unique=args.unique or False,
            sub_data=int(args.sub_data) if args.sub_data else None,
            batch_size=int(args.batch_size) if args.batch_size else 128)


if __name__ == "__main__":
    main(sys.argv[1:])
