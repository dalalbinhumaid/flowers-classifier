import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

#function to resize and normalize an image
def process_images(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image,( 224, 224))
    image /= 255
    return image.numpy()

#function to predict the top k porbabilities alongside their classes
def predict(image_path, model, top_k = 1):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_images(image)
    image = np.expand_dims(image,  axis=0)
    predictions = model.predict(image)
    
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k= top_k)
    
    return top_k_values.numpy(), top_k_indices.numpy()
 
 #function to load a json file into a variable   
def json_extractor(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names
 
  #function to convert integer classes into categorical names
def convert_class_names(classes, class_names):
    names = []
    for i in classes[0]:
        names.append(class_names[str(i+1)])
    return names