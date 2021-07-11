import argparse
import tensorflow as tf
import tensorflow_hub as hub
from utils import *


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Python script that allows you to predict an image via a pre-trained model.')
    
    #positional arguments
    parser.add_argument('image_path')
    parser.add_argument('model')
    
    #optional arguments
    parser.add_argument('--top_k', type = int, default = 1)
    parser.add_argument('--category_names')
    
    args = parser.parse_args()
    
    #reload the pre trained model and predict the image
    reloaded_model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})
    probs, classes = predict(args.image_path, reloaded_model, args.top_k)
    
    #extract the class_names if they are supplied
    if args.category_names is not None:
        class_names = json_extractor(args.category_names)
        classes = convert_class_names(classes, class_names)
   
    print(probs)
    print(classes)
    
    
        
    

 
