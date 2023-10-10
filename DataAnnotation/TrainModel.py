import tensorflow as tf
import onnx
import cv2
import numpy as np
import os
import PlotTraining

from Dataloader import loadDataset
import CnnModel

model = CnnModel.getEncoderModel()
modelName = "testModel"
testing = False
epochs = 5

# Create training and test datasets
print("Dataset loading")
X_train, X_test, y_train, y_test = loadDataset("TestDataset - Kopi.csv")

trainImg = []
testImg = []

print("readying train")
for img in X_train:
    bgrImg = cv2.imread(img)
    trainImg.append(cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB))

print("readying test")
for img in X_test:
    bgrImg = cv2.imread(img)
    testImg.append(cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB))

# Train model
print("Start training")
history = model.fit(np.asarray(trainImg), y_train, epochs=epochs, batch_size=128, validation_data=(np.asarray(testImg), y_test))

# Test model
print("testing model")
test_loss, test_acc = model.evaluate(np.asarray(testImg), y_test)
print('\nTest accuracy: {}'.format(test_acc))

model.save(modelName + ".keras")

os.system('python -m tf2onnx.convert --saved-model tmp_model --output "' + modelName + '.onnx"')

# Convert model to lite
converter = tf.lite.TFLiteConverter.from_saved_model("tmp_model")
tflite_model = converter.convert()
tf.io.write_file("Models", tflite_model)


Y_pred = model.predict(np.asarray(testImg))
y_pred = np.argmax(Y_pred, axis=1)

PlotTraining.print_confusion_matrix(y_test, y_pred)
PlotTraining.plot_metric(history, 'loss')
PlotTraining.plot_metric(history, 'accuracy')