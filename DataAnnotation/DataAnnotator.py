import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

filename = "TestDataset.csv"
path = "D:/outdoor/"
DatasetPath = "NewDataset/"
files_and_directories = os.listdir(path)
showPlots = False
showImg = False


def findAverage(img):
    average_color_row = np.average(img, axis=0)
    average_color = np.average(average_color_row, axis=0)
    # print(average_color)
    d_img = np.ones((50, 50, 3), dtype=np.uint8)
    d_img[:, :] = average_color
    imgList.append(d_img)

    avgHSV = cv.cvtColor(d_img.copy(), cv.COLOR_BGR2HSV)

    hueList.append(avgHSV[0][0][0] * 2)
    valueList.append(translate(avgHSV[0][0][2], 0, 255, 0, 100))

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

hueList = []
valueList = []
imgList = []
fileNames = []

# Create dataset
# Load all folders
for folder in files_and_directories:
    if (not os.path.exists(DatasetPath + folder)):
        os.mkdir(DatasetPath + folder)
    # Load all files in a given folder
    for file in os.listdir(path + folder + "/"):
        cropped = []
        imageName = Path(file).stem
        img = cv.imread(path + folder + "/" + file)

        height, width, channels = img.shape
        newWidth = height/1.71

        copy = img.copy()
        crop1 = copy[0:height, 0:int(newWidth)]
        crop2 = copy[0:height, width - int(newWidth):width]
        cv.waitKey()

        cropped.append(cv.resize(crop1, [256, 256]))
        cropped.append(cv.resize(crop2, [256, 256]))

        if showImg:
            cv.imshow('image', img)
            cv.imshow('crop 1', cropped[0])
            cv.imshow('crop 2', cropped[1])
            cv.waitKey()

        for x, crop in enumerate(cropped):
            cv.imwrite(DatasetPath + folder + '/' + imageName + str(x) + '.png', crop)
            findAverage(crop)

        '''histg = cv.calcHist([cv.cvtColor(resizedImg, cv.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
        red_hist = cv.calcHist([resizedImg], [2], None, [256], [0, 255])
        green_hist = cv.calcHist([resizedImg], [1], None, [256], [0, 255])
        blue_hist = cv.calcHist([resizedImg], [0], None, [256], [0, 255])
        if showPlots:
            plt.subplot(4, 1, 1)
            plt.plot(histg)
            plt.title("gray")

            plt.subplot(4, 1, 2)
            plt.plot(red_hist)
            plt.title("red")

            plt.subplot(4, 1, 3)
            plt.plot(green_hist)
            plt.title("green")

            plt.subplot(4, 1, 4)
            plt.plot(blue_hist)
            plt.title("blue")

            plt.tight_layout()
            plt.show()'''
        '''cv.imshow('Origin image', img)
        cv.imshow('Average Color', d_img)
        cv.waitKey()
        key = input("Class")
        file = open(filename, "a")
        file.write(fileName + ";" + str(key) + "\n")
        file.close()
        print(key)'''
    
#Annotate dataset
'''for folder in os.listdir(DatasetPath):
    # Load all files in a given folder
    for file in os.listdir(DatasetPath + folder):
        img = cv.imread(DatasetPath + folder + '/' + file)
        cv.imshow(str(file), img)
    cv.waitKey()
    cv.destroyAllWindows()

    key = input("Class")
    csvFile = open(filename, "a")
    for file in os.listdir(DatasetPath + folder):
        csvFile.write(path + folder + '/' + file + ';' + str(key) + '\n')
    csvFile.close()
'''


plt.hist(hueList)
plt.xlabel('hue')
plt.ylabel('hue count')
plt.show()

plt.xlim(0, 360)
plt.ylim(0, 100)
for x, img in enumerate(imgList):
    #cv.imshow('img', img)
    #cv.waitKey()
    print(img[0][0])
    print(hueList[x])
    plt.imshow(img, extent=[hueList[x] - 5, hueList[x] + 5, valueList[x] + 5, valueList[x] - 5])
plt.show()
