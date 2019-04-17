# %load_ext tensorboard.notebook

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# the following has been removed in tf2.0
# tf.logging.set_verbosity(tf.logging.ERROR)

# set up tensorboard log directory
logdir="tf_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(logdir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# need the following to successfully run. 
# Otherwise there will be cuDNN error in training 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

# define dataloader and preprocessing
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, 256, 256)
    image = tf.image.random_crop(image, (224, 224, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = image/255  # normalize to [0,1] range
    return image
# path = "/home/nan/sneaker_ai/screened_data/air_jordan_1/air_jordan_1_362.png"
# plt.imshow(load_and_preprocess_image(path))

# set up datasets
from sklearn.utils import shuffle
train_df = pd.read_csv("train_df.csv")
train_df = shuffle(train_df)
train_image_paths = train_df.loc[:, "Image_File"].tolist()
train_image_labels = train_df.loc[:, "Label"].tolist()

test_df = pd.read_csv("test_df.csv")
# test_df = shuffle(test_df)
test_image_paths = test_df.loc[:, "Image_File"].tolist()
test_image_labels = test_df.loc[:, "Label"].tolist()
# covertion between class names and model outputs
label_names = ["AJ "+str(i) for i in range(1,24)]
label_to_index = dict((name, index) 
                      for index, name in enumerate(label_names))

train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function 
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

train_ds = train_ds.map(load_and_preprocess_from_path_label)
test_ds = test_ds.map(load_and_preprocess_from_path_label)


# specify batch size and buffering for datasets
BATCH_SIZE = 16

# # Setting a shuffle buffer size as large as the dataset ensures that the data is
# # completely shuffled.
# train_ds = train_ds.apply(
#  tf.data.experimental.shuffle_and_repeat(buffer_size=len(train_df)))
train_ds = train_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=200))
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#test_ds = test_ds.apply(
#  tf.data.experimental.shuffle_and_repeat(buffer_size=len(test_df)))
test_ds = test_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=200))
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# import mobilenet v2 model
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224,224, 3), include_top=False)
# mobile_net.trainable=False
mobile_net.trainable=True


# MobileNetV2: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
# imagenet.utils: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
# mobile net expects prepocessed input ranging from -1 to 1
def change_range(image,label):
    return 2*image-1, label

train_ds = train_ds.map(change_range)
test_ds = test_ds.map(change_range)

# add fc layers after mobilenet for transfer learning
model = tf.keras.Sequential([
    mobile_net,
    # take an average of each channel so that the model woks with different input sizes
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation = tf.nn.relu),
#     tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
#     tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_names), activation=tf.nn.softmax)])

# configure model optimizer, loss
model.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-6), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

print(model.summary())

# %tensorboard --logdir {logdir}

# model training
num_epochs = 100
training_history = model.fit(train_ds, epochs=num_epochs, 
                             validation_data=test_ds,
                             callbacks=[tensorboard_callback])
