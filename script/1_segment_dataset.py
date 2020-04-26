# extract parts from dataset of images with COCO annotation format

import sys, os, re
import json
import argparse
import glob
import pathlib
import shutil

import numpy as np
from PIL import Image
import yaml

from codae.processing import extract_part_from_polygons


def parse():

    parser = argparse.ArgumentParser(
        description='Segment dataset of images into parts. Accept COCO annotation file as input.')

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--dataset_name', type=str, default="modanet")

    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    if args.dataset_name != "deepfashion2"\
        and args.dataset_name != "modanet"\
        and args.dataset_name != "imaterialist":
            raise Exception("Provided dataset not supported (deepfashion2/modanet/imaterialist)")


    # open annotation file & dataset info ######################################

    if args.dataset_name == "deepfashion2":

        annotation_path = os.path.join( args.dataset_path, "train_coco_annotation.json")

        if not os.path.exists(annotation_path):

            print("Consolidating deepfashion2 to COCO annotation format ...")
            from codae.dataset import df2_to_coco
            df2_to_coco(args.dataset_path)
            print("... done.")

        with open(annotation_path, 'r') as f:
            COCO_annotation = json.load(f)

        info_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "../codae/dataset/deepfashion2.yaml" )

        with open(info_path, 'r') as stream:
            try:
                info = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise e

    if args.dataset_name == "modanet":
        try:
            with open(args.dataset_path + "TODO", 'r') as f:
                COCO_annotation = json.load(f)
        except:
            raise Exception("Annotatio file for Modanet not found")

        info_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "../codae/dataset/modanet.yaml" )

        with open(info_path, 'r') as stream:
            try:
                info = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise e

    if args.dataset_name == "imaterialist":
        try:
            with open(args.dataset_path + "TODO", 'r') as f:
                COCO_annotation = json.load(f)
        except:
            raise Exception("Annotation file for Imaterialist not found")

        info_path = os.path.join(
            pathlib.Path(__file__).parent.absolute(),
            "../codae/dataset/imaterialist.yaml" )

        with open(info_path, 'r') as stream:
            try:
                info = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise e

    # empty output directory
    choice = input("Erase output directory ? (y/Y/n/N)")
    if choice == "y" or choice == "Y":
        print("Erasing output directory ...")
        shutil.rmtree(args.output_path, ignore_errors=True) 
        print("... done.")

    # create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    # prepare annotations ######################################################

    # index annotations per image ID
    annotation_index = {}
    for ann in COCO_annotation["annotations"]:
        if not ann["image_id"] in annotation_index:
            annotation_index[ann["image_id"]] = []
        annotation_index[ann["image_id"]].append( ann )

    # index category by ID
    category_index = {}
    for cat in COCO_annotation["categories"]:
        category_index[cat["id"]] = cat["name"]


    # process images ###########################################################

    for image_section in COCO_annotation["images"]: # for all image in json

        image_id = image_section["id"] # get image ID

        # make sure output surb dir exists
        output_sub_dir = os.path.join(args.output_path, str(image_id))
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # open image
        image = Image.open(
            os.path.join(
                args.dataset_path,
                info["IMAGE_PATH"],
                image_section["file_name"]))

        part_id = 0
        for ann in annotation_index[image_id]:

            print("Segmenting image ID %d" %image_id, end="\r")

            image_size = image.size

            try: # extract part from polygon

                p = ann["segmentation"][0]

                # from (x1, y1, x2, y2, ...) to ((x1, y1), (x2, y2), ...)
                p = [[p[i*2], p[(i*2)+1]] for i in range(int(len(p)/2))]

                extracted_part = extract_part_from_polygons( image, [p], crop=True )

            except:
                print("Error while extracting parts from polygons, image ID %d." %image_id)

            m = np.mean(extracted_part)
            if m < 0.01: # filter black images
                print("Encountered black image mean %f. Ignoring ..." %m)
                continue

            part_id_str = str(part_id).zfill(2)

            category_id = ann["category_id"]

            output_file_name = str(image_id) + "_" + part_id_str + "_" + category_index[ann["category_id"]].replace(" ", "_") + ".jpg"

            try: 
                # export extracted part to disk in sub folder
                Image.fromarray(extracted_part).save(
                    os.path.join(output_sub_dir, output_file_name))
                part_id += 1
            except:
                print("Error while exporting image ID %d to disk." %image_id)
    
    print("... done.")