
import sys, os, re
import json
import argparse
import glob

from PIL import Image
import yaml

from codae.dataset import extract_part_from_polygons


def parse():

    parser = argparse.ArgumentParser(
        description='Segment dataset of images into parts. Accept COCO annotation file as input.')

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--dataset_name', type=str, default="modanet")

    parser.add_argument('--dataset_path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    if args.dataset != "deepfashion2"\
        and args.dataset != "modanet"\
        and args.dataset != "imaterialist":
            raise Exception("Provided dataset not supported (deepfashion2/modanet/imaterialist)")


    # open annotation file & dataset info ######################################

    if args.dataset_name == "deepfashion2":

        if not os.path.exists("annotation.json"):

            print("Consolidating deepfashion2 to COCO annotation format ...")
            from codae.dataset import df2_to_coco
            df2_to_coco()
            print("... done.")

        with open(args.dataset_path, 'r') as f:
            COCO_annotation = json.load(f)

        info = yaml.load(
            os.path.dirname(__file__) + "../codae/dataset/deepfashion2.yaml")

    if args.dataset_name == "modanet":
        try:
            with open(args.dataset_path + "TODO", 'r') as f:
                    COCO_annotation = json.load(f)
        except:
            raise Exception("Annotatio file for Modanet not found")

        info = yaml.load(
            os.path.dirname(__file__) + "../codae/dataset/modanet.yaml")

    if args.dataset_name == "imaterialist":
        try:
            with open(args.dataset_path + "TODO", 'r') as f:
                    COCO_annotation = json.load(f)
        except:
            raise Exception("Annotation file for Imaterialist not found")

        info = yaml.load(
            os.path.dirname(__file__) + "../codae/dataset/imaterialist.yaml")

    # create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    # process images ###########################################################

    # list all image in directory
    image_list = glob.glob(os.path.join(
        info["IMAGE_PATH"], "*.jpg"),
        recursive=True)

    # get annotation dict
    annotation = glob.glob( os.path.join(
        info["ANNOTATION_FILE"],
        "*.json"),
        recursive=True)

    r = re.compile("item[0-9]+")
    for image_path in image_list:

        # get image ID
        image_id = 










   
    annotation = {}
    r = re.compile("item[0-9]+")
    for annotation_file in annotation_list: # for all image/annotation

        image_id = annotation_file.split(".")[0].split("/")[-1]

        # open annotation file
        with open(annotation_file) as f:
            annotation[image_id] = json.load(f)
        annotation[image_id]["item"] = {}

        # make sure output surb dir exists
        output_sub_dir = os.path.join(args.output_path, image_id)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # load image
        image = Image.open(os.path.join(args.image_path, image_id+".jpg"))

        # for each annotated item in the image
        part_id = 0
        for item in list(filter(r.match, annotation[image_id].keys())):

            print("Segmenting image ID %s." %image_id)

            # from (x1, y1, x2, y2, ...) to ((x1, y1), (x2, y2), ...)
            # only first segmentation mask is selected (full segmentation)
            seg = annotation[image_id][item]["segmentation"][0]
            polygon = [[seg[i*2], seg[(i*2)+1]] for i in range(int(len(seg)/2))]

            # extract part from polygon
            try:
                extracted_part = extract_part_from_polygons( image, [polygon] )
            except:
                print("Error while extracting parts from polygons, image ID %s." %image_id)

            part_id_str = str(part_id).zfill(2)

            output_file_name = image_id+"_"+part_id_str+"_"+annotation[image_id][item]["category_name"].replace(" ", "_")+".jpg"

            try: 
                # export extracted part to disk in sub folder
                Image.fromarray(extracted_part).save(
                    os.path.join(output_sub_dir, output_file_name))

                # rename "item%d" key to "part_id" key
                annotation[image_id]["item"][part_id_str] = annotation[image_id][item]
                annotation[image_id].pop(item, None)

                part_id += 1
            except:
                print("Error while exporting image ID %s to disk." %image_id)

    # export modified annotation file to disk
    with open(os.path.join(args.output_path, "annotation_seg.json"), "w+") as f:
        f.write( json.dumps(annotation) )
    
    print("DONE")