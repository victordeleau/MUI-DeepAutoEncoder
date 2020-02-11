
import sys, os
import json

from PIL import Image


def parse():

    parser = argparse.ArgumentParser(
        description='Encode dataset of image.')

    parser.add_argument('--image_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--sub_dir_scan', type=bool, default=True)

    parser.add_argument('--model', type=str, default="resnet50")

    parser.add_argument('--annotation_s1', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.model == "resnet18":
        model = models.resnet18(pretrained=True)
    elif args.model == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise Exception("Model name not found (resnet18/resnet50).")

    # list all image in directory
    image_list = glob.glob(args.image_path, recursive=args.sub_dir_scan)

    # open annotation_seg
    with open(args.annotation_s1, "r") as f:
        annotation_s1 = json.load(f)

    # instantiate feature extractor model
    feature_extractor = FeatureExtractor("resnet50")

    result = {}

    for image_id in annotation_s1.keys(): # for all images

        for part_id in annotation_s1[image_id]["item"]: # for all parts

            # read image part
            

            # extract feature vector
            image_vector = feature_extractor.encode(image)