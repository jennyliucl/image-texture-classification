import tensorflow as tf
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
import pandas as pd

from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from keras.utils.np_utils import *
from imblearn.over_sampling import SMOTE

train = pd.read_csv("D:/Machine Learning/AOI/AOI/3/3/AOI_train.csv")

# Model / data parameters
input_shape = (128, 128, 1)

# the data, split between train and test sets

model = keras.Sequential(
    [
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(6, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 50


x_train = np.array(X_data).reshape((1500,128*128))
x_res, y_res = SMOTE(random_state=42).fit_resample(x_train, train["Label"])
x_train = np.array(x_res).reshape((2370,128,128,1))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train,y_res, epochs=epochs,batch_size=batch_size, validation_split=0.2)


score = model.evaluate(x_train[1301:1500], train["Label"][1301:1500])
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# predict data
test = pd.read_csv("D:/Machine Learning/AOI/AOI/3/3/AOI_test.csv")
X_data_test = []
for i in range(0,len(test)):
  img = cv2.imread(os.path.join("D:/Machine Learning/AOI/AOI/3/3/AOI_Test_Image", test.loc[i,"ID"]))
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
  X_data_test.append(image/255)
x_test = np.array(X_data_test).reshape((1000, 128,  128,1))

result=[]
result = model.predict(x_test)
result_2=[]
for i in range(0,len(result)):
    temp = np.argmax(result[i])
    result_2.append(temp)

dataframe = pd.DataFrame(result_2)
dataframe.columns=["result_cnn"]
dataframe.to_csv('D:/Machine Learning/AOI/AOI/3/3/result_cnn3.csv')