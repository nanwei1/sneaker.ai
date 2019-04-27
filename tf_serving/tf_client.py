from __future__ import print_function

import base64
import requests
import json
import tensorflow as tf
import numpy as np

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/sneaker_ai:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = '/home/nan/sneaker_ai/tf_serving/air_jordan_1_3.png'


def main():
  # Download the image
  # dl_request = requests.get(IMAGE_URL, stream=True)
  # dl_request.raise_for_status()

  # Compose a JSON Predict request (send JPEG image in base64).
  # jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
  # predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
  # image_data = skimage.io.imread(IMAGE_URL)
  image = tf.io.read_file(IMAGE_URL)
  image = tf.image.decode_jpeg(image, channels=3)
  image = image/255  # normalize to [0,1] 
  image = tf.image.resize_with_pad(image, 224, 224)
  image = np.expand_dims(image.numpy(), axis=0)
  print(image.shape)
  predict_request = json.dumps({'instances': image.tolist()})
  # print(predict_request)
  

  # Send few requests to warm-up the model.
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    print(response.text)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]

  max_ind = np.argmax(prediction)
  predicted_result = 'Air Jordan ' + str(max_ind+1)
  print('Prediction class: {}, avg latency: {} ms'.format(
      predicted_result, (total_time*1000)/num_requests))


if __name__ == '__main__':
  main()

