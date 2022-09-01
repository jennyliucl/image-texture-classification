from PIL import Image
from numpy import asarray
import pandas as pd
import numpy as np

import cv2
import os
train = pd.read_csv("D:/Machine Learning/AOI/AOI/3/3/AOI_train.csv")

# load the image

# convert image to numpy array
X_data = []
for i in range(0,len(train)):
  img = cv2.imread(os.path.join("D:/Machine Learning/AOI/AOI/3/3/Train_Image", train.loc[i,"ID"]))
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  image = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
  X_data.append(image/255)


  #img = cv2.imread(os.path.join("D:/Machine Learning/AOI/AOI/3/3/Train_Image", train.loc[5,"ID"]))
  #image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
  #cv2.imshow("l",img)
  #cv2.waitKey(0)


