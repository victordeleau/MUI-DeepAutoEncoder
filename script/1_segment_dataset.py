
import sys, os, re
import json
import argparse
import glob

from PIL import Image

from codae.dataset import extract_part_from_polygons


def parse():

    parser = argparse.ArgumentParser(
        description='Segment dataset of images into parts.')

    parser.add_argument('--image_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--annotation_path', type=str, required=True)

    parser.add_argument('--sub_dir_scan', type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    # list all image in directory
    image_list = glob.glob(os.path.join(args.image_path, "*.jpg"), recursive=args.sub_dir_scan)

    # get annotation dict
    annotation_list = glob.glob(os.path.join(args.annotation_path, "*.json"), recursive=args.sub_dir_scan)
   
    annotation = {}
    r = re.compile("item*")
    for annotation_file in annotation_list:

        image_id = annotation_file.split(".")[0].split("/")[-1]

        with open(annotation_file) as f:
            annotation[image_id] = json.load(f)

        output_sub_dir = os.path.join(args.output_path, image_id)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # load image
        image = Image.open(os.path.join(args.image_path, image_id+".jpg"))

        # for each annotated item in the image
        for item in list(filter(r.match, annotation[image_id].keys())):

            print("Segmenting image ID %s." %image_id)

            # from (x1, y1, x2, y2, ...) to ((x1, y1), (x2, y2), ...)
            # only first segmentation mask is selected (full)
            seg = annotation[image_id][item]["segmentation"][0]
            polygon = [[seg[i*2], seg[(i*2)+1]] for i in range(int(len(seg)/2))]

            # extract part from polygon
            extracted_part = extract_part_from_polygons( image, [polygon] )

            output_file_name = image_id+"_"+annotation[image_id][item]["category_name"].replace(" ", "_")+".jpg"

            # export extracted part to disk in sub folder
            try:
                Image.fromarray(extracted_part).save(
                    os.path.join(output_sub_dir, output_file_name))
            except:
                print("Error while exporting image ID %s to disk." %image_id)
    
    print("DONE")