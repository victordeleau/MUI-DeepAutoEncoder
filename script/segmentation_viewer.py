# display segmentation result from 1_segment_dataset.py

import argparse
import os
from PIL import ImageShow, Image


def parse():

    parser = argparse.ArgumentParser(
        description='Segmentation viewer.')

    parser.add_argument('--segmentation_path', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()

    class FehViewer(ImageShow.UnixViewer):
        def show_file(self, filename, **options):
            os.system('feh %s' % filename)
            return 1

    ImageShow.register(FehViewer, order=-1)

    # loop over segmentation directories
    for subdir, dirs, files in os.walk(args.segmentation_path):

        if len(files) != 0:
            print("\nObservation ID %s" %(subdir.split("/")[-1]))

        # loop over the image it contains
        for file in files:

            image_path = os.path.join(subdir, file) 

            print("Part category = %s" %(file.split(".")[0].split("_")[-1]), end="\r")

            with open(image_path, 'r') as f:
                Image.open(image_path).show()

            input()