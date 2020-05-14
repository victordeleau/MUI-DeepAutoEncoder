
from copy import copy

from skimage import draw
from PIL import Image
from skimage import draw
import cv2
import numpy as np


def extract_part_from_bbox(image, bbox):
    """
    crop image using bounding box [x,y,width,height]
    input
        image : PIL.Image
        bbox : list
    """

    return np.array(image)[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3], :]


def extract_part_from_polygons(image, polygons, crop=False):
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
    if crop:
        extracted_part = extracted_part[
            int(bbox[1]):int(bbox[3]),
            int(bbox[0]):int(bbox[2]),:]

    return extracted_part