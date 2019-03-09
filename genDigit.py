from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import models
import numpy as np
import os
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import model_from_json
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

model = models.Sequential()

my_file = Path("genDigit.json")
if my_file.is_file():
    json_file = open('genDigit.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("genDigit.h5")
    print("Loaded model from disk")
else:
    print('creating')
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    del train
    X_train /= 255.0
    test /= 255.0
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)
    Y_train = to_categorical(Y_train, num_classes=10)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=64, epochs=2, validation_data=(X_val, Y_val), verbose=2)
    model.save_weights("genDigit.h5")
    json_model = model.to_json()
    with open("genDigit.json", "w") as json_file:
        json_file.write(json_model)
    print("saved")


myMap = dict()
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for id, name in enumerate(files):
            if name.endswith(format):
                fullName = os.path.join(root, name)
                myMap[id] = fullName
                fileList.append(fullName)
    return fileList

myFileList = createFileList('images/')
predicts = np.zeros((len(myFileList), 28, 28, 1))
for id, file in enumerate(myFileList):
    img_file = Image.open(file)
    img_grey = img_file.convert('L')

    value = np.asarray(img_grey, dtype='float32').reshape((28, 28, 1))
    value = value.flatten()
    value = np.reshape(value, (28, 28, 1))
    predicts[id] = value


predicts_value = model.predict(predicts, batch_size=4)

for id, entry in enumerate(predicts_value):
    print(myMap[id], '  ', np.argmax(entry))



