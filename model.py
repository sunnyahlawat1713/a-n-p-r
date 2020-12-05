import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract
import re

def find_number_plate(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 11, 90, 90)

    image_edges = cv2.Canny(gray, 30, 200)

    contours, new = cv2. findContours(image_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    copy_of_image = image.copy()

    _ = cv2.drawContours(copy_of_image, contours, -1, (255, 0, 255), 2)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    plate = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(edges_count) == 4:
            x, y, w, h = cv2.boundingRect(c)
            plate_loc = image[y:y +h, x:x+w]
            break

    cv2.imwrite('C:/Users/Sunny Ahlawat/Desktop/stack_fusion/plate_images/plate.png', plate_loc)

    pytesseract.pytesseract.tesseract_cmd =  '/app/.apt/usr/bin/tesseract'
    #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(plate_loc)



    return re.sub('[^a-zA-Z0-9]+', '', text)
