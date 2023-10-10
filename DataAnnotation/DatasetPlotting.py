import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import os
import cv2
import math

DatasetPath = "NewDataset/"
avgHSVList = []
def findAverage(img):
    average_color_row = np.average(img, axis=0)
    average_color = np.average(average_color_row, axis=0)
    # print(average_color)
    d_img = np.ones((50, 50, 3), dtype=np.uint8)
    d_img[:, :] = average_color
    avgHSV = cv2.cvtColor(d_img.copy(), cv2.COLOR_BGR2HSV)
    avgHSVList.append(avgHSV)
    avgHue = avgHSV[0][0][0]
    avgValue = translate(avgHSV[0][0][2], 0, 255, 0, 100)

    return round(avgHue), round(avgValue), cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def roundup(x):
    return int(math.ceil(x / 10.0))

print("loading images")
avgHueList = []
avgValueList = []
avgRGBList = []
for folder in os.listdir(DatasetPath):
    for file in os.listdir(DatasetPath + folder):
        img = cv2.imread(DatasetPath + folder + "/" + file)
        avgHue, avgValue, avgRGB = findAverage(img)
        avgHueList.append(avgHue)
        avgValueList.append(avgValue)
        avgRGBList.append(avgRGB)


dataList = list(zip(avgHueList,avgValueList))
npArray = np.zeros(shape=(180, 100))
print("Creating numpy array")
for x in range(len(avgHueList)):
    npArray[avgHueList[x]-1][avgValueList[x]-1] += 1

print("plotting heatmap")
ax = sns.heatmap(npArray.T)
plt.show()

print("plotting img graph")
plt.xlim(0, 180)
plt.ylim(0, 100)
for x, img in enumerate(avgRGBList):
    #cv2.imshow('img', img)
    #print(img)
    #print(avgHSVList[x])
    #print(avgHueList[x] - 5, avgHueList[x] + 5, avgValueList[x] + 5, avgValueList[x] - 5)
    #cv2.waitKey()
    plt.imshow(img, extent=[avgHueList[x] - 5, avgHueList[x] + 5, avgValueList[x] + 5, avgValueList[x] - 5])
plt.show()