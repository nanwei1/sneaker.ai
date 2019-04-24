from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import random
import os
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import skimage

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/map')
def map():
    return render_template('map.html', title='Map')

@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
   f.save(path)
   result_image = run_inference(path)
   return render_template('uploaded.html', title='Success', user_image=f.filename, result_iamge=result_image)

@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')



###########################
# load model and weights
def Model():
    weights_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'model_weights.ckpt')
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224,224, 3), include_top=False)
    # add fc layers after mobilenet for transfer learning
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation = tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation = tf.nn.relu),
        tf.keras.layers.Dense(23, activation=tf.nn.softmax)])
    model.load_weights(weights_dir)
    return model

# pre-process image similar to those in training
def img_preprocessing(path):
    image = skimage.io.imread(path)
    # convert grayscale images to rgb
    if image.ndim ==2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    H, W = image.shape[0], image.shape[1]
    if H>W:
        H_new, W_new = 224, round(W/H*224)
    else:
        H_new, W_new = round(H/W*224), 224
    image = skimage.transform.resize(image, output_shape=(H_new, W_new), preserve_range=True)
    # zero-pad the image
    temp = np.zeros((224,224,3))
    temp[:image.shape[0], :image.shape[1], :] = image
    image = temp
    image = image/255   # to range [0, 1]
    image = 2*image-1    # to range[-1,1]
    return image

def logits_to_probs(logits):
    probs = np.exp(logits) / np.exp(logits).sum()
    probs = probs.reshape(probs.shape[1],)
    return probs

# plot pie chart showing probability and save
def plot_probs(probs):
    indices = np.argsort(probs)[::-1]
    probs = np.sort(probs)[::-1]
    top_k = 3
    probs_temp = np.append(probs[:top_k],np.sum(probs[top_k:]))
    keys = ['AJ '+str(i+1) for i in indices[:top_k]] + ['Others']
    explode = (0.1, 0.05, 0.05, 0)
    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.pie(probs_temp, explode=explode, labels=keys, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # print(app.config['UPLOAD_FOLDER'])
    # print(os.path.dirname(app.config['UPLOAD_FOLDER']))
    # print(os.path.join(os.path.dirname(app.config['UPLOAD_FOLDER']), 'result/result.png'))
    result_img = datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
    result_img_path = os.path.join(os.path.dirname(app.config['UPLOAD_FOLDER']), 'result',result_img)
    plt.savefig(result_img_path)
    return result_img

def run_inference(image_path):
    # need the following to successfully run.
    # Otherwise there will be cuDNN error in training
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.as_default()
    image = img_preprocessing(image_path)
    model = Model()
    logits = model.predict(np.expand_dims(image, axis=0))
    probs = logits_to_probs(logits)
    return plot_probs(probs)
