
#libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential

from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

#importing dataset
from datasets import load_dataset

sets = load_dataset("vikhyatk/synthetic-pepe")

#taking nopepe dataset
df = tf.keras.utils.image_dataset_from_directory('drive')
df = tf.keras.utils.image_dataset_from_directory('drive', batch_size=len(df))
data_iterator = df.as_numpy_iterator()

dataset1 = data_iterator.next()

fig, ax = plt.subplots(ncols=5, figsize=(20,20))
for idx, img in enumerate(sets[0][:5]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(sets[1][idx])

#scaling
df = df.map(lambda x,y: (x/255, y))
df.as_numpy_iterator().next()
len(df)

#splitting
train_size = int(len(df)*0.7)
val_size = int(len(df)*0.2)
test_size = int(len(df)*0.1)

train = df.take(train_size)
val = df.skip(train_size).take(val_size)
test = df.skip(train_size+val_size).take(test_size)

train

#Building CNN
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()


#Training CNN
logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

#Plotting loss function and accuracy
IMG = plt.figure()
plt.plot(hist.history['loss'], color='BLUE', label='loss')
plt.plot(hist.history['val_loss'], color='RED', label='val_loss')
IMG.suptitle('Loss Function', fontsize=35)
plt.legend(loc="upper left")
plt.show()


IMG = plt.figure()
plt.plot(hist.history['accuracy'], color='BLUE', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='RED', label='val_accuracy')
IMG.suptitle('Accuracy', fontsize=35)
plt.legend(loc="upper left")
plt.show()


#Evaluation
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())

#Testing
import cv2

img = cv2.imread('/content/drive/MyDrive/photo collage/adi bhai p1.png')
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
ybar = model.predict(np.expand_dims(resize/255, 0))

ybar

if ybar > 0.5:
    print(f'Pepe')
else:
    print(f'Not Pepe')

import cv2
img = cv2.imread('/content/drive/MyDrive/synthetic-pepe/026dedb7-54c2-483b-a368-ae11874080fe.webp')
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

ybar = model.predict(np.expand_dims(resize/255, 0))

ybar

if ybar < 0.5:
    print(f'Pepe')
else:
    print(f'Not Pepe')
