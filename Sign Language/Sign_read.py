import datetime

import cv2
import numpy as np
from PIL import ImageGrab
import time
import os
def capture_screen():
    screenshot = ImageGrab.grab()
    #turn into opencv readable form
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np,cv2.COLOR_RGB2BGR)

    return screenshot_np
def showMatches(template_directory,threshold = 0.9):
    #capture the sign the user signs
    screenshot = capture_screen()
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    #go through every alphabet sign template and find the best match
    for filename in os.listdir(template_directory):
        if filename.endswith('.png'):  # Filter for image files
            template_path = os.path.join(template_directory, filename)
            template = cv2.imread(template_path)
            temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = temp_gray.shape[:2]

            matches = cv2.matchTemplate(gray_screenshot, temp_gray, cv2.TM_CCOEFF_NORMED)
            locations = np.where(matches >= threshold)

        if locations[0].size != 0:
            print(f"match found:{filename}")
            break #stop searching after finds the matching alphabet
def sign_read():
    showMatches("/Users/tanzihao/PycharmProjects/auto/Sign Language/sign_templates",threshold=0.77)
    time.sleep(0.1)
def main():
    while True:
        sign_read()
main()