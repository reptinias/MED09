import cv2
import os
import math
import keyboard
import matplotlib.pyplot as plt

datasetPath = "resizedDataset/"
n = 0
fileName = "TestDataset.csv"

for folder in os.listdir(datasetPath):
    folderList = os.listdir(datasetPath + folder)
    skipFolder = False
    imgList = []
    for file in folderList:
        img = cv2.imread(datasetPath + folder + "/" + file)
        imgList.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.figure(figsize=(16, 8))
    columns = 20
    for i, image in enumerate(imgList):
        plt.subplot(int(len(imgList)/columns) + 1, columns, i + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()

    '''while(len(folderList) > 0):
        imgList = []
        if(len(folderList) > 20):
            n = 20
        else:
            n = len(folderList)
        for i in range(n):
            img = cv2.imread(datasetPath + folder + "/" + folderList[i])
            imgList.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #cv2.imshow(str(folderList[i]), img)

        fig, axs = plt.subplots(1, n, figsize=(10, 5))
        for x, img in enumerate(imgList):
            axs[x].imshow(img)
            axs[x].axis('off')
        plt.show()

        while True:
            if keyboard.read_key() == "a":
                skipFolder = True
                break
            if keyboard.read_key() == "space":
                break

        if skipFolder == True:
            break
        folderList = folderList[n:]
'''
    key = input("Class")
    csvFile = open(fileName, "a")
    for file in os.listdir(datasetPath + folder):
        csvFile.write(datasetPath + folder + '/' + file + ';' + str(key) + '\n')
    csvFile.close()