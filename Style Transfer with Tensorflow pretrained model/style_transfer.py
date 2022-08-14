import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


content_image = load_image('path_of_content') # the content image whose style will be changed
style_image = load_image('path_of_style') # the image that will be used as the style

stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0] # output image

plt.imshow(np.squeeze(stylized_image))
plt.show()
