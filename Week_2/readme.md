# Week 2 Notes

##### Learning Objectives
- Use callback functions for tracking model loss and accuracy during training
- Make predictions on how the layer size affects network predictions and training speed
- Implement pixel value normalization to speed up network training
- Build a multilayer neural network for classifying the Fashion MNIST image dataset

##### Exploring how to use data
Machine Learning depends on having good data to train a system with. To explore the data.
To see how to load that data and prepare it for training.

##### The structure of Fashion MNIST data

Using a number is a first step in avoiding bias -- instead of labelling it with words in a specific language and excluding people who don’t speak that language!

You can learn more about bias and techniques to avoid it here. https://ai.google/responsibilities/responsible-ai-practices/

##### New concepts
Flatten: Flatten just takes that square and turns it into a 1-dimensional array.
Dense: Adds a layer of neurons
Activation function: Each layer of neurons need an activation function to tell them what to pass
ReLU: it only passes values 0 or greater to the next layer in the network.
Softmax takes a list of values and scales these so the sum of all elements will be equal to 1. When applied to model outputs, you can think of the scaled values as the probability for that class.