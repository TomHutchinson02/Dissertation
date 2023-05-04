#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import keras
from keras.layers import BatchNormalization
import tensorflow as tf
from keras import regularizers
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models, regularizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from tensorflow import keras
from keras.losses import sparse_categorical_crossentropy
import sys
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, Callback
from PIL import Image
from keras.metrics import Precision, Recall
sys.modules['Image'] = Image 


# In[2]:


target_size = 64,64
Batch_size = 128
learning_rate = 0.003
epochs = 170
SEED = 12


# In[3]:


def step_decay(epoch):
    drop = 0.5       # factor by which the learning rate is reduced
    epochs_drop = 2  # number of epochs after which to drop the learning rate
    lr = learning_rate * drop**((epoch+1)//epochs_drop)  # calculate the updated learning rate
    return lr

lr_scheduler = LearningRateScheduler(step_decay)


# In[4]:


# Define a lambda function for downsampling
downsample_fn = lambda image: image if train_generator.classes[train_generator.index_array[train_generator.batch_index-1]] != 7 or tf.random.uniform(shape=(), minval=0, maxval=1) < 0.5 else None
datagen = ImageDataGenerator()

datagen = ImageDataGenerator(

    validation_split=0.2
) # set validation split

train_generator = datagen.flow_from_directory(
    'C:/Users/TomHu/OneDrive/Desktop/Final Year project/Train',
    target_size=target_size,
    batch_size=Batch_size,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle =True,
    color_mode='rgb'  # set color_mode to 'rgb' for converting images to RGB format
)

validation_generator = datagen.flow_from_directory(
    'C:/Users/TomHu/OneDrive/Desktop/Final Year project/Train', # same directory as training data
    target_size=target_size,
    batch_size=Batch_size,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle =True,
    color_mode='rgb'  # set color_mode to 'rgb' for converting images to RGB format
) # set as validation data


test_generator = datagen.flow_from_directory(
    'C:/Users/TomHu/OneDrive/Desktop/Final Year project/Test',
    target_size=target_size,
    batch_size=Batch_size,
    class_mode='categorical',
    seed=SEED,
    shuffle =True,
    color_mode='rgb'  # set color_mode to 'rgb' for converting images to RGB format
)


# In[5]:


from keras import backend as K

def f1_macro(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[22]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation="relu", input_shape=(64, 64, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"))
model.add(Dropout(0.6))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation="relu", strides=(2,2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), strides=(2,2)))

model.add(Conv2D(1024, kernel_size=(3, 3), padding='same', activation="relu", strides=(2,2)))
model.add(Dropout(0.6))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation="relu", strides=(2,2)))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation="relu", strides=(2,2)))

model.add(Flatten())



model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(64, activation="relu")) 
model.add(Dropout(0.2))
model.add(Dense(14, activation="sigmoid"))

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.000001)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy", Precision(), Recall(), f1_macro])

model.summary()


# In[15]:


class_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0.04, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
class_weights = {'Abuse': 1, 'Arrest': 1, 'Arson': 1, 'Assault': 1, 'Burglary': 1, 'Explosion': 1, 'Fighting': 1, 'NormalVideos': 0.075,
                 'RoadAccidents': 1, 'Robbery': 1, 'Shooting': 1, 'Shoplifting': 1, 'Stealing': 1, 'Vandalism': 1}
# Get the number of samples in each class
class_counts = train_generator.classes
class_labels = list(train_generator.class_indices.keys())
class_sample_counts = np.bincount(class_counts)
class_sample_dict = dict(zip(class_labels, class_sample_counts))
count = 0
# Calculate the weighted count of samples in each class
weighted_class_sample_dict = {} 
for count, (class_label, sample_count) in enumerate(class_sample_dict.items()):
    class_label_str = class_labels[count]
    weight = class_weights[class_label_str]
    weighted_sample_count = weight * sample_count
    weighted_class_sample_dict[class_label_str] = weighted_sample_count
    count=+1


# Print out the number of samples in each class with applied class weights
print("Class Sample Counts (Weighted):")
for class_label, sample_count in weighted_class_sample_dict.items():
    print("Class {}: {} samples".format(class_label, sample_count))


# In[24]:


model.fit(
        train_generator,
        steps_per_epoch= (train_generator.n/Batch_size)/200,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps= (validation_generator.n/Batch_size)/200,
        callbacks=[early_stop, reduce_lr],
        batch_size=Batch_size,
        class_weight=class_dict
)


# In[17]:


model.evaluate(test_generator, batch_size=Batch_size)


# In[ ]:


print(train_generator.class_indices)


# In[ ]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# 
