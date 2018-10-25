from network.yolo_v0 import YoloV0
import argparse
import os
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', action='store', type=str, help='Path to .cfg file.')
    parser.add_argument('-pp', action='store', type=str, help='path to parameters that should be '
                                                              'loaded', default=None)
    parser.add_argument('-epochs', action='store', type=int, help='Number of epochs to process.', default=10)
    parser.add_argument('-timages', action='store', type=str, help='Path to training images.',
                        default="E:\Andrew\Dataset\Training set\Images")
    parser.add_argument('-vimages', action='store', type=str, help='Path to validation images.', default=None)
    parser.add_argument('-tlabels', action='store', type=str, help='Path to training labels.',
                        default="E:\Andrew\Dataset\Training set\Annotations")
    parser.add_argument('-vlabels', action='store', type=str, help='Path to validation labels.',
                        default=None)
    parser.add_argument('-summstep', action='store', type=int, help='Specifies number of steps after summaries will be'
                                                                    ' written', default=10)
    parser.add_argument('-test', action='store_true')

    arguments = parser.parse_args()
    print(arguments)

    if not os.path.isfile(arguments.cfg):
        parser.error("argument -cfg: invalid path to .cfg file.")
    if not os.path.exists(arguments.timages):
        parser.error("argument -timages: invalid path to training images directory")
    if not os.path.exists(arguments.tlabels):
        parser.error("argument -tlabels: invalid path to training labels directory")
    if arguments.vimages is not None:
        if not os.path.exists(arguments.vimages):
            parser.error("argument -vimages: invalid path to validation images directory")
    if arguments.vlabels is not None:
        if not os.path.exists(arguments.vlabels):
            parser.error("argument -vlabels: invalid path to validation labels directory")

    return arguments


def main():
    cfg = None
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]) and os.path.isfile(sys.argv[1]):
            with open(sys.argv[1]) as file:
                cfg = json.load(file)
        else:
            print('Path should point to existing file')
            exit(1)
    else:
        print('Enter path to train_cfg file as first argument')
        exit(1)
    net = YoloV0(cfg['net_cfg'])
    if cfg['weights'] is not None:
        net.restore(path=cfg['weights'])
    net.optimize(cfg['training_set'], cfg['validation_set'], cfg['parameters']['num_epochs'], cfg['parameters'])


if __name__ == '__main__':
    main()
