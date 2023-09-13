import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


filename = "TestDataset.csv"
path = "TestDataset/"
files_and_directories = os.listdir(path)
showPlots = False
showImg = False

'''img = cv.imread(path + files_and_directories[0])
cv.imshow('image', img)
cv.waitKey()
value = input()
print(value)

if not os.path.exists(filename):
    open(filename, "w").close()
'''
hueList = []
for fileName in files_and_directories:
    img = cv.imread(path + fileName)
    resizedImg = cv.resize(img, [255, 255])

    if showImg:
        cv.imshow('image', img)
        cv.imshow('image', resizedImg)

    histg = cv.calcHist([cv.cvtColor(resizedImg, cv.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
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
        plt.show()

    average_color_row = np.average(img, axis=0)
    average_color = np.average(average_color_row, axis=0)
    #print(average_color)
    d_img = np.ones((10, 10, 3), dtype=np.uint8)
    d_img[:, :] = average_color

    avgHue = cv.cvtColor(d_img, cv.COLOR_BGR2HSV)
    print(avgHue[0][0][0]*2)
    hueList.append(avgHue[0][0][0]*2)

    '''cv.imshow('Origin image', img)
    cv.imshow('Average Color', d_img)
    cv.waitKey()
    key = input("Class")
    file = open(filename, "a")
    file.write(fileName + ";" + str(key) + "\n")
    file.close()
    print(key)'''

plt.hist(hueList)
plt.xlabel('hue')
plt.ylabel('hue count')
plt.show()