from network.yolo_v0 import YoloV0
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', action='store', type=str, help='Path to .cfg file.')
    parser.add_argument('-pp', action='store', dest='weights_path', type=str, help='path to parameters that should be '
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
    args = parse_args()
    training_set = [args.timages, args.tlabels]
    valid_set = None
    net = YoloV0(args.cfg)
    if (args.vimages is not None) and (args.vlabels is not None):
        valid_set = [args.vimages, args.vlabels]
    # restore if path to weights was provided
    if args.pp is not None:
        net.restore(path=args.pp)
    net.optimize(args.epochs, training_set, valid_set, args.summstep, do_test=args.test)


if __name__ == '__main__':
    main()
