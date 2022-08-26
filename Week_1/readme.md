# Week 1 Notes

##### Learning Objectives
- Monitor the accuracy of the housing price predictions
- Analyze housing price predictions that come from a single layer neural network
- Use TensorFlow to build a single layer neural network for fitting linear models 

##### Introduction: A conversation with Andrew Ng

Name of the Lecture: Laurence Moroney

You will need these packages if you will run the notebooks locally:

tensorflow==2.7.0
scikit-learn==1.0.1
pandas==1.1.5
matplotlib==3.2.2
seaborn==0.11.2

##### Get started with Google Colaboratory (Coding TensorFlow)

- built-in code snippets EX: visualization
- shared with google drive
- To learn Colab: google research seedbank

##### C1_W1_Lab_1_hello_world_nn.ipynb

Keras: framework

Sequential: class of Keras

Dense: type of layers

Define and compile the neural network

compile: to tell the computer how to calculate the model

compile: specify loss and optimizer

The **loss function** measures the guessed answers against the known correct answers and measures how well or how badly it did.

It then uses the **optimizer function** to make another guess. Based on how the loss function went, it will try to minimize the loss.

training the model: it 'learns' the relationship between the x's and y's

You can use the model.predict() method to have it figure out the y for a previously unknown x.

neural networks deal with probabilities. 

So given the data that we fed the model with, it calculated that there is a very high probability that the relationship between x and y is y=2x-1, but with only 6 data points we can't know for sure.

##### Week 1 Resources

- AI For Everyone https://www.deeplearning.ai/ai-for-everyone/
- video updates from the TensorFlow team youtube.com/tensorflow
- Play with a neural network right in the browser(The spiral is particularly challenging!) http://playground.tensorflow.org