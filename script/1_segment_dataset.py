
import sys, os
import json
import glob

class DatasetSegmenter(object):
    """
    A class to segment a dataset of images into contained images.
    """

    def __init__(self, output_folder):

        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


    def from_polygons(self, image_path, annotation_path):
        """
        extract parts from a set of images using provided polygons
        input
            image_path: str
                path of image folder on disk
            annotation_path: str
                path of json annotation file on disk
        """

        # get list of image
        image_list = glob.glob(image_path)

        # get annotation dict
        with open() as f:
            annotation = json.load(f)

        # for all images
        for image_name in image_list:

            image_name = image_name.split(".")[0]

            # create sub directory for image
            os.makedirs(image_name)

            # for all apparel in image
            for _ in annotation[]:

                # extract cropped and masked apparel

                # write to disk

                pass


    def from_mask_rcnn(self, image_path):
        """
        extract parts from a set of images using mask rcnn model
        input
            image_path: str
                path of image folder on disk
        """

        pass