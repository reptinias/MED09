from CnnModel import getFullNet
from Dataloader import loadDataset
import numpy as np
import cv2
import PlotTraining
import tensorflow as tf
import matplotlib.pyplot as plt
import onnx
import tf2onnx

modelName = "MultiModelNetwork"
testing = False
epochs = 5

# Create training and test datasets
print("Dataset loading")
X_train, X_test, y_train, y_test = loadDataset("TestDataset.csv")

trainImg = []
testImg = []

print("readying train")
for img in X_train:
    try:
        bgrImg = cv2.imread(img)
        trainImg.append(cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB))
    except:
        print("Could not find image")


print("readying test")
for img in X_test:
    try:
        bgrImg = cv2.imread(img)
        testImg.append(cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB))
    except:
        print("Could not find image")

print(len(trainImg), len(testImg))
model = getFullNet()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
modelHis = model.fit(np.asarray(trainImg), y_train, epochs=epochs, batch_size=32, validation_data=(np.asarray(testImg), y_test))
test_loss, test_acc = model.evaluate(np.asarray(testImg), y_test)

model.save(modelName + ".keras")

input_signature = [tf.TensorSpec([None, 256, 256, 3], tf.float32, name='x')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save_model(onnx_model, modelName + ".onnx")

# Convert model to lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(modelName + '.tflite', 'wb') as f:
  f.write(tflite_model)

Y_pred = model.predict(np.asarray(testImg))
y_pred = np.argmax(Y_pred, axis=1)

PlotTraining.print_confusion_matrix(y_test, y_pred)
PlotTraining.plot_metric(modelHis, 'loss')
PlotTraining.plot_metric(modelHis, 'accuracy')

def getWrongPreds(predicted, trueValues, testData):
    print(predicted, trueValues)
    for x, value in enumerate(trueValues):
        if value != predicted[x]:
            wrongPreds.append(np.asarray(testData[x]))

wrongPreds = []
getWrongPreds(y_pred, y_test, np.asarray(testImg))

plt.figure(figsize=(16, 8))
columns = 10
for i, image in enumerate(wrongPreds):
    plt.subplot(int(len(wrongPreds) / columns) + 1, columns, i + 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
plt.show()
