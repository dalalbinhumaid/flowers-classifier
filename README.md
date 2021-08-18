# flowers-classifier
## Deep Learning
## Project: Image Classifier

This project is divided to two parts the first part implements an image classifier with TensorFlow using a neural network and the second is a  Python script that runs from the command line in which an image path is entered, and some optional parameter and the script will predict the image. 
> The project dataset is Oxford of 102 flowers categories it can be found through [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) 

#### Usage 
```console 
$ python predict.py ./test_images/orchid.jpg my_model.h5
```
#### Optional - Return the top 3 most likely classes 
```console 
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
```
#### Optional - Use a `label_map.json` file to map labels to flower names
```console 
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
```


### Part 1 - Developing an Image Classifier with Deep Learning
1. **Data Exploration**: Upon loading the dataset I explored its contents and features then plotted an image and created a pipeline to normalize each image.
2. **Building and Training the Classifier**: It was required that the MobileNet pre-trained network is loaded first from TensorFlow Hub. Then I trained the network, plotted the loss and accuracy values achieved during training for the training and validation set. Lastly, saved the trained model as a Keras model.
3. **Inference for Classification**: I Wrote a function that uses my trained network for inference. The function is called `predict` that takes an image, a model, and then returns the top _k_ most likely class labels along with the probabilities. 

If `top_k=5` the output of the `predict` function should be something like :

```python
probs, classes = predict(image_path, model, 5)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

The `predict` function will also need to handle pre-processing, which I wrote a separate function `process_image` that performs that.

### Part 2 - Building the Command Line Application
Using Python `argsparse module` to get command line input, then predict the image by reloading the pre-trained model labeled as `model.h5`. The file `utils.py` contains all the helper functions I used to pre-process and predict the image along with the optional arguments functions.
