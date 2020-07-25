# encode polyvore dataset

import json
import yaml
import argparse
import os
import pathlib

from PIL import Image

from codae.processing import FeatureExtractor

def parse():

    parser = argparse.ArgumentParser(
        description='Encode Polyvore.')

    parser.add_argument('--image_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--model', type=str, default="resnet50")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    # load info file
    info_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "../codae/dataset/polyvore.yaml" )
    with open(info_path, 'r') as stream:
        try:
            info = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise e

    # load annotations
    with open(info["ANNOTATION_FILE"], 'r') as f:
        annot = json.load(f)

    # revert merge
    merge = {}
    for dest in info["CATEGORY"].items():
        for origin in dest[1]:
            merge[origin] = dest[0]

    # invoque feature extractor
    fe = FeatureExtractor(args.model)

    output = {}

    # for all outfits
    for outfit in annot:

        # for all items in outfit
        for item in outfit["items"]:

            try:

                # retrieve new category
                dest_cat = merge[item["categoryid"]]

                if not outfit["set_id"] in output.keys():
                    output[outfit["set_id"]] = {}

                # retrieve image
                image = Image.open(os.path.join(args.image_path,str(outfit["set_id"]),str(item["index"]) + ".jpg"))

                # encode corresponding image
                output[outfit["set_id"]][dest_cat] = fe.encode(image).tolist()

            except:
                    #print("ai")
                    continue

    # only keep outfit with at least 3 categories

    print(output)

    # write file to disk
    with open(os.path.join(args.output_path, "polyvore_encoded.json"), 'w+') as f:
        json.dump(output, f)