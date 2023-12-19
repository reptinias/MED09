import cv2
import os
import numpy as np

filename = "NewDatasetFilenames.csv"
imagePath = "Dataset/"

for imageName in os.listdir(imagePath):
    image = cv2.imread(imagePath + imageName)
    resized = cv2.resize(image, [290, 580])

    cv2.imshow("img", resized)
    cv2.waitKey()
    cv2.destroyAllWindows()
    key = input("Class")

    file = open(filename, "a")
    file.write(imagePath + imageName + ";" + str(key) + "\n")
    file.close()

