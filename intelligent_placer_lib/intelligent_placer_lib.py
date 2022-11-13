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
IS_DEBUG_MODE = False
CONST_ROWS, CONST_COLS = 1024, 1024

CLOSING_TIMES = 3
BLACK_MASKS_THRESOLD = [
    np.array((0, 0, 80), np.uint8),
    np.array((182, 120, 255), np.uint8),
]

# some unushuall thresolds for some current items
UNSUAL_THRESOLDS = [
    [
        np.array((26, 45, 0), np.uint8),
        np.array((78, 255, 255), np.uint8),
    ]  # for object 5
]

# Thresolds to contours
SHORT_CONTOUR = 30
NEAR_BORDER = 0.08


def debug_plot_img(img):
    if IS_DEBUG_MODE:
        plt.imshow(img, cmap="gray")
        plt.show()
    return


# Function which will load and preprocess inputed image
def load_and_preproc_image(path_to_png_jpg_image_on_local_computer: str) -> np.ndarray:
    img = cv2.imread(path_to_png_jpg_image_on_local_computer)
    img = cv2.GaussianBlur(
        img, ksize=(5, 5), sigmaX=5
    )  # Maybe make this numbers adaptive by image size?
    img = cv2.resize(img, (CONST_COLS, CONST_ROWS))
    return img


# Functions which trying to detect elements, which will be black after HSV-inRange binarization
def detect_black_elements(img):
    img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh_img = cv2.inRange(img_tmp, BLACK_MASKS_THRESOLD[0], BLACK_MASKS_THRESOLD[1])
    # debug_plot_img(thresh_img)
    thresh_img.dtype = np.bool8
    res_img = ~thresh_img
    return res_img


# Function which trying to detect some unisuall objects, which are not simillar like other
def detect_unusual_elements(img):
    img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    res_img = np.full(
        shape=(img_tmp.shape[0], img_tmp.shape[1]), fill_value=False, dtype="bool"
    )

    for thresold in UNSUAL_THRESOLDS:
        thresh_img = cv2.inRange(img_tmp, thresold[0], thresold[1])
        thresh_img.dtype = np.bool8
        res_img = thresh_img | res_img

    return res_img


def get_contours(img, orig_img):
    img.dtype = np.uint8
    img = img * 255
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Filter contours
    res_contours = []
    for i in range(len(contours)):
        # if contour inside some other contour -> leave
        if hierarchy[0][i][3] != -1:
            continue

        # if contour is too short -> leave (there could be some little noizes)
        M = cv2.moments(contours[i])
        cont_len = M["m00"]
        if cont_len < SHORT_CONTOUR:
            continue

        # if contour near picture sides -> leave (there is artifacts after photo with flash)
        # TODO : Maybe try other params to thresholding to delete this?
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if (
            (cX < CONST_ROWS * NEAR_BORDER)
            or (cX > CONST_ROWS * (1.0 - NEAR_BORDER))
            or (cY < CONST_COLS * NEAR_BORDER)
            or (cY > CONST_COLS * (1.0 - NEAR_BORDER))
        ):
            continue

        # append contour
        res_contours.append(contours[i])

    # Trying found polygon contour
    # TODO : Maybe make some more robust criteria? Maybe mean value from mask?
    polygon_contour_id = -1
    max_int = -1
    for i in range(len(res_contours)):
        mask = np.zeros(shape=img.shape, dtype=np.uint8)
        cv2.drawContours(
            mask, res_contours, i, color=(255, 255, 255), thickness=cv2.FILLED
        )
        mean = cv2.mean(orig_img, mask)
        new_int = np.linalg.norm(mean, 2)
        if new_int > max_int:
            max_int = new_int
            polygon_contour_id = i

    return res_contours, polygon_contour_id


# Function which provides detecting list, objects from image
def detect_elements(img) -> Tuple[np.ndarray, int]:
    # detecting all objects
    res_img = detect_black_elements(img)
    res_img = res_img | detect_unusual_elements(img)

    # Clear some holes in objects
    res_img.dtype = np.uint8
    res_img = res_img * 255
    for _ in range(CLOSING_TIMES):
        res_img = binary_closing(res_img, footprint=np.ones((13, 13)))
    debug_plot_img(res_img)

    # Detect contours and polygon contour
    contours, p_id = get_contours(res_img, img)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 4)
    cv2.drawContours(img, contours, p_id, (0, 0, 255), 2)
    debug_plot_img(img)

    return (contours, p_id)


# Function which will return polygon mask and array of objects' masks
def generate_masks(img, elements_contours, polygon_id) -> Tuple[np.ndarray, np.ndarray]:
    objects = []
    polygon = None
    for i in range(len(elements_contours)):
        mask = np.zeros(shape=img.shape, dtype=np.uint8)
        cv2.drawContours(
            mask, elements_contours, i, color=(255, 255, 255), thickness=cv2.FILLED
        )

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(mask)
        region_masked = np.zeros(shape=(h, w), dtype=np.uint8)
        region_masked[:, :] = mask[y : y + h, x : x + w]

        if i == polygon_id:
            polygon = region_masked
        else:
            objects.append(region_masked)

        if IS_DEBUG_MODE:
            debug_printed_mask = np.zeros(
                shape=(region_masked.shape[0], region_masked.shape[1], 3)
            )
            channel = 2 if i == polygon_id else 1
            debug_printed_mask[:, :, channel] = region_masked[:, :]
            cv2.cvtColor(region_masked, cv2.COLOR_GRAY2BGR)
            debug_plot_img(debug_printed_mask)
    return (objects, polygon)


# Function which will provide tests with masks to make decidition: is it possible to place objects in polygon
def preprocessing_tests(objects_masks: np.ndarray, polygon: np.ndarray) -> bool:
    # First (and, at the moment, only) test - square test
    polygon_pixels = cv2.countNonZero(polygon)
    objs_pxls = 0
    for obj_masks in objects_masks:
        objs_pxls += cv2.countNonZero(obj_masks)
    if objs_pxls > polygon_pixels:
        return False
    else:
        return True


# Function which will try to place object in polygon
def is_possible_to_place(ploygon: np.ndarray, objects: np.ndarray) -> bool:
    # dummy return
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
    TEST_PATH = "tests/4_0.jpg"
    check_image(TEST_PATH)
