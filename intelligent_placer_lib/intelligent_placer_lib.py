# imports
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgba2rgb
from skimage.data import page
from skimage.feature import canny
from skimage.filters import (
    gaussian,
    sobel,
    threshold_local,
    threshold_minimum,
    threshold_otsu,
    try_all_threshold,
)
from skimage.measure import label as sk_measure_label
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_erosion, binary_opening

# Debugging constants and functions
IS_DEBUG_MODE = True


def debug_plot_img(img):
    if IS_DEBUG_MODE:
        plt.imshow(img, cmap="gray")
        plt.show()
    return


# Function which will load and preprocess inputed image
def load_and_preproc_image(path_to_png_jpg_image_on_local_computer: str) -> np.ndarray:
    CONST_ROWS, CONST_COLS = 1024, 1024
    img = cv2.imread(path_to_png_jpg_image_on_local_computer)
    img = cv2.GaussianBlur(
        img, ksize=(5, 5), sigmaX=5
    )  # Maybe make this numbers adaptive by image size?
    img = cv2.resize(img, (CONST_COLS, CONST_ROWS))
    return img


# Function which provides detecting list, objects from image
def detect_elements(img) -> Tuple[np.ndarray, np.ndarray]:
    # TODO : Если все объекты разом не выделяются, то может сделать кучу раз объединение по идеальным параметрам для сложно выделяемых объектов?
    HSV_MIN = np.array((0, 0, 92), np.uint8)
    HSV_MAX = np.array((255, 100, 255), np.uint8)

    # Thresolding process
    img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh_img = cv2.inRange(img_tmp, HSV_MIN, HSV_MAX)
    debug_plot_img(thresh_img)
    res_img = np.abs(np.ones_like(thresh_img) * 255 - thresh_img)

    # Closing micro-holes of objects
    CLOSING_TIMES = 5
    for _ in range(CLOSING_TIMES):
        res_img = binary_closing(res_img, footprint=np.ones((13, 13)))
    debug_plot_img(res_img)

    # Take counters of masks
    res_img.dtype = np.uint8
    contours, _ = cv2.findContours(res_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # TODO : подумать, как убрать дефекты побокам, мб выбрать другой принцип трезолдинга?

    img_contours = np.uint8(np.zeros((img_tmp.shape[0], img_tmp.shape[1])))
    cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 1)
    # debug_plot_img(img_contours)
    cv2.imshow("lines", img_contours)

    # заглушка
    return (np.zeros(shape=(1, 1)), 1)


# Function which will return polygon mask and array of objects' masks
def generate_masks(img, elements, polygon_id) -> Tuple[np.ndarray, np.ndarray]:
    # заглушка
    return (np.zeros(shape=(1, 1)), np.zeros(shape=(1, 1)))


# Function which will provide tests with masks to make decidition: is it possible to place objects in polygon
def preprocessing_tests(objects_masks: np.ndarray, polygon: np.ndarray) -> bool:
    # заглушка
    return False


# Function which will try to place object in polygon
def is_possible_to_place(ploygon: np.ndarray, objects: np.ndarray) -> bool:
    # заглушка
    return False


# Function which provides main project executions
def check_image(path_to_png_jpg_image_on_local_computer: str):
    # step 0 - load image and scale it to some size
    img = load_and_preproc_image(path_to_png_jpg_image_on_local_computer)

    # step 1 - detect elements and polygon index in elements
    elements_contours, polygon_id = detect_elements(img)

    # step 2 - generate masks
    objects, polygon = generate_masks(img, elements_contours, polygon_id)

    # step 3 - preprocessing criteries
    pre_tests_result = preprocessing_tests(objects, polygon)

    # step 4 - main algorithm
    if pre_tests_result == True:
        return is_possible_to_place(polygon, objects)
    else:
        return False


# For debuging
if __name__ == "__main__":
    TEST_PATH = "tests/1_0.jpg"
    check_image(TEST_PATH)
