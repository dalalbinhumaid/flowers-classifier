# flowers-classifier
## Deep Learning
## Project: Image Classifier

This project is divided to two parts; the first part implements an image classifier with TensorFlow

[project dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories.

### Part 1: Developing an Image Classifier with Deep Learning
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


