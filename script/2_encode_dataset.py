import sys, os
import json
import glob
import argparse

from PIL import Image

from codae.processing import FeatureExtractor


def parse():

    parser = argparse.ArgumentParser(
        description='Encode dataset of image.')

    parser.add_argument('--image_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--sub_dir_scan', type=bool, default=True)

    parser.add_argument('--model', type=str, default="resnet18")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.model != "resnet18" and args.model != "resnet50":
        raise Exception("Model name not found (resnet18/resnet50).")

    # invoque feature extractor
    fe = FeatureExtractor(args.model)

    # list all image in directory
    image_path_list = glob.glob(
        os.path.join(args.image_path, "*/*.*"),
        recursive=args.sub_dir_scan)

    nb_error = 0

    print("Processing ...")

    # encode images in dictionnary
    output = {}
    for image_path in image_path_list:

        try:

            # get image/part ID
            file_name = image_path.split("/")[-1].split(".")[-2]
            image_id = file_name.split("_")[0]
            part_id = file_name.split("_")[1]
            category = " ".join( file_name.split("_")[2:] )

            # read image
            part_image = Image.open(image_path)

            # add to dictionnary
            if not image_id in output.keys():
                output[image_id] = {}
            output[image_id][category] = fe.encode(part_image).tolist()

        except Exception as e:
            nb_error += 1
            continue

    print("... done.")
    print("%d errors encountered." %nb_error)

    # export dictionnary to file
    with open(os.path.join(args.output_path, "encoded_parts.json"), "w+") as f:
        f.write( json.dumps(output) )