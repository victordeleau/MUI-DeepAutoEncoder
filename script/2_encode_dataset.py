
import sys, os
import json

import torchvision.models as models

def parse():

    parser = argparse.ArgumentParser(
        description='Encode dataset of image.')

    parser.add_argument('--image_path', type=str)

    parser.add_argument('--output_path', type=str)

    parser.add_argument('--sub_dir_scan', type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.model == "resnet18":
        model = models.resnet18()
    elif args.model == "resnet50":
        model = models.resnet50()
    else:
        raise Exception("Model name not found (resnet18/resnet50).")

    # list all image in directory
    image_list = glob.glob(args.image_path, recursive=args.sub_dir_scan)

    result = {}

    # for all images
    for image_name.split(".")[0] in image_list:

        pass