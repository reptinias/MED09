import cv2
import os
from pathlib import Path

datasetPath = "NewDataset/"
newDatasetPath = "resizedDataset/"

if (not os.path.exists(newDatasetPath)):
    os.mkdir(newDatasetPath)

for folder in os.listdir(datasetPath):
    if (not os.path.exists(newDatasetPath + folder)):
        os.mkdir(newDatasetPath + folder)
    for file in os.listdir(datasetPath + folder):
        img = cv2.imread(datasetPath + folder + "/" + file)
        resizedImg = cv2.resize(img, [256, 256])
        imageName = Path(file).stem
        cv2.imwrite(newDatasetPath + folder + '/' + imageName + '.png', resizedImg)