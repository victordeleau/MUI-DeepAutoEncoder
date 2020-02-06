
from copy import copy

from skimage import draw
from PIL import Image
from skimage import draw
import cv2
import numpy as np


def extract_parts_from_mask(image, mask_image, labels, boxes):

    """
    Extract each detected parts (crop + blackened background)
    in an image using masks and boxes.

    arguments:
        image: rgb opencv image.
            The image in which parts should be extracted.
        mask_image: grayscale (!) opencv image.
            The masked image used to mask the background of extracted parts.
        labels: list of predicted class index.
        boxes: list of of [x1,y1,x2,y2] bounding boxes.

    return:
        list of extracted parts as opencv images in boxes order.
    """

    extracted_parts = []

    for i in range(len(boxes)):

        # apply mask to cropped image
        shape = image.shape
        part_masked = copy(image).reshape([shape[0]*shape[1], 3])
        part_masked[ np.where((mask_image.flatten() != labels[i])) ] = [0, 0, 0]
        part_masked = part_masked.reshape([shape[0], shape[1], 3])

        # crop using bounding box
        part_masked_cropped = part_masked[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2]), :]

        extracted_parts.append(\
            cv2.cvtColor(part_masked_cropped, cv2.COLOR_BGR2RGB))

    return extracted_parts



def extract_part_from_polygons(image, polygons):
    """
    extract each part in the image using a list of polygons
    input
        image: as a numpy array
        polygons: list of polygon (each being a list of [ [x1, y1], ... ] point)
    output
        image as a numpy array
    """

    image = np.array(image)
    polygons = np.array(polygons)
    mask = np.zeros((image.shape[0], image.shape[1])).astype(np.bool)

    for polygon in polygons: # build mask from polygons
        try: 
            m = np.zeros((image.shape[0], image.shape[1]))
            cv2.fillConvexPoly(m, polygon, 1)
            mask = np.add( m.astype(np.bool), mask)
        except:
            pass # polygon might be invalid

    # apply mask to image
    extracted_part = np.zeros_like(image)
    extracted_part[mask] = image[mask]

    # flatten outer dimension and extract bbox from polygons
    polygon = polygons.reshape(-1, polygons.shape[-1])
    bbox = [
        min([p[0] for p in polygon]),
        min([p[1] for p in polygon]),
        max([p[0] for p in polygon]),
        max([p[1] for p in polygon])]

    # crop extracted part according to bbox
    return extracted_part[
                int(bbox[1]):int(bbox[3]),
                int(bbox[0]):int(bbox[2]),:]