from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing import image
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

TRAIN_DIR = 'D:\\ML\\Task 4 - Copy\\Task 4 - Copy\\images\\train'
TEST_DIR = 'D:\\ML\\Task 4 - Copy\\Task 4 - Copy\\images\\test'

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_paths,labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

print(train)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

print(test)
print(test['image'])

from tqdm.notebook import tqdm

def extract_features(images):
    features = []
    for image_path in tqdm(images):
        img = load_img(image_path, color_mode='grayscale')
        img_array = img_to_array(img)
        features.append(img_array)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features = extract_features(train['image'])

test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train,num_classes = 7)
y_test = to_categorical(y_test,num_classes = 7)

model = Sequential()

#Adding Convolutional Layers
model.add(Conv2D(128, kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam' , loss = 'categorical_crossentropy', metrics = 'accuracy')

model.fit(x = x_train, y = y_train, batch_size = 128, epochs=100, validation_data = (x_test,y_test)) 

model_json = model.to_json()
with open("Personality_Prediction.json",'w') as json_file:
    json_file.write(model_json)
model.save("Personality_Prediction.h5")

from keras.models import model_from_json

json_file = open("Personality_Prediction.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("Personality_Prediction.h5")

label = ['angry','disgust','fear','happy','neutral','sad','surprise']

def ef(image):
    img = load_img(image, color_mode='grayscale')
    feature = img_to_array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

image = 'images/train/angry/27.jpg'
print("Original image is angry")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("Model prediction is : ",pred_label) 

import matplotlib.pyplot as plt
plt.imshow(img.reshape(48,48),cmap='gray')