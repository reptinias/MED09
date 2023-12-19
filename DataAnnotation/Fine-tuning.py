from Dataloader import loadDataset
import PlotTraining
import cv2
import tensorflow as tf
import numpy as np

modelName = "MultiModelNetwork"
epochs = 5

# Create training and test datasets
print("Dataset loading")
X_train, X_test, y_train, y_test = loadDataset("NewDatasetFilenames.csv")

trainImg = []
testImg = []

print("readying train")
for img in X_train:
    try:
        bgrImg = cv2.imread(img)
        trainImg.append(cv2.resize(cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB),[256,256]))
    except:
        print("Could not find image")


print("readying test")
for img in X_test:
    try:
        bgrImg = cv2.imread(img)
        testImg.append(cv2.resize(cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB),[256,256]))
    except:
        print("Could not find image")

model = tf.keras.models.load_model('MultiModelNets/MultiModelNetwork.keras')
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelHis = model.fit(np.asarray(trainImg), y_train, epochs=20, initial_epoch=5, batch_size=32, validation_data=(np.asarray(testImg), y_test))

Y_pred = model.predict(np.asarray(testImg))
y_pred = np.argmax(Y_pred, axis=1)

PlotTraining.print_confusion_matrix(y_test, y_pred)
PlotTraining.plot_metric(modelHis, 'loss')
PlotTraining.plot_metric(modelHis, 'accuracy')