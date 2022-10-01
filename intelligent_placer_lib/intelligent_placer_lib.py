#####
#
#   ТЕСТОВАЯ ОТРАБОТКА ИДЕЙ ДЛЯ РЕАЛИЗАЦИИ ПЛАНА АЛГОРИТМА
#   УВАЖИТЕЛЬНО ПРОШУ ПРОВЕРЯЮЩИХ НЕ РАСМАТРИВАТЬ ЕГО В КАЧЕСТВЕ СДАЧИ РАБОТЫ "ПЛАН АЛГОРИТМА"
#
#####

# imports
from matplotlib import pyplot as plt
import numpy as np
from imageio import imread, imsave
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.filters import sobel, gaussian, threshold_local, try_all_threshold, threshold_otsu, threshold_minimum
from skimage.data import page
from skimage.morphology import binary_opening, binary_closing, binary_erosion
from skimage.measure import regionprops
from skimage.measure import label as sk_measure_label
import cv2

# Some global variables
IS_DEBUG_MODE = True

# Function which provides detecting list, objects from image
def detect_elements(img):
    if IS_DEBUG_MODE:
        fig, ax = try_all_threshold(img, figsize=(15, 15), verbose=False)
        plt.show()

    img_otsu = threshold_minimum(img, nbins=64)
    res_otsu = img <= img_otsu

    if IS_DEBUG_MODE:
        plt.imshow(res_otsu, cmap='gray')
        plt.show()
    return [], 0

# Function which provides main project executions
def check_image(path_to_png_jpg_image_on_local_computer : str):
    # step 0 - load image
    img = imread(path_to_png_jpg_image_on_local_computer)
    img = rgb2gray(img)

    #if IS_DEBUG_MODE:
    #    plt.imshow(img, cmap="gray")
    #    plt.show()

    # step 1 - detect elements and polygon index
    elements, polygon_id = detect_elements(img)

    # step 2 - generate masks 

    # step 3 - preprocessing criteries

    # step 4 - main algorithm

    return

# For debuging
if __name__ == "__main__":
    TEST_PATH = "tests/4_0.jpg"
    check_image(TEST_PATH)