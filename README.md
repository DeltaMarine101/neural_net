 ![](https://img.shields.io/badge/version-alpha%201.0-red.svg) ![](https://img.shields.io/badge/python-3.71-green.svg) ![](https://img.shields.io/badge/licence-MIT-blue.svg) 

# Welcome to my neural net!

Hi! I'm Jeremie and this is my first attempt at making a **neural network** from scratch. If you want to learn about neural networks yourself, you can [watch these videos](https://www.youtube.com/watch?v=aircAruvnKk&list=PLjtrFat0jG9e8K-A_QpIJAzsBHTgPARQW). If you want to play around with making one yourself, feel free to edit, tinker and destroy my code!


## Setup

This code was written in [Python 3.71](https://www.python.org/downloads/release/python-371/), and requires numpy and matplotlib. Simply run the following commands to install them:
```
pip install numpy
pip install matplotlib
```
You will need to clone this repo (rather than download the ZIP) if you want it to work with the MNIST files provided, since they are managed by github lfs (large file system).
	

## Usage

Start by creating a neural_net instance and pass in a shape for the network. The shape is a tuple with each value representing the number of nodes in that layer, including both input and output layers. 
```
# This takes in 28 by 28 images, passes them through 2 hidden layers of 16 nodes, then outputs 10 values
nn = neural_network((28 * 28, 16, 16, 10))
```
The neural net is now initialised with random weights and biases, so in order to train it you can back-propogate training data. Make sure you only pass in batches of 10 to 100 training examples, you can re-run backprop as many times as you want and it will continue to improve.
```
# This will backproagate the network on the first 100 training examples
nn.backprop(training_data[:100])

# This will backproagate the network on the next 100 training examples
nn.backprop(training_data[100:200])
```
You can run the network with a single example.
```
# show=True will print the result of the final layer
nn.run(image, show=True)
```
You can test the performance of the network using accuracy() or loss().
```
# This will print the average loss over the first 200 training examples
print("Loss:", nn.loss(training_data[:200]))

# This will print the accuracy over the testing data set
print("Accuracy:", str(nn.test(testing_data) * 100) + "%")
```


## Visualization

You might be wondering why you needed to install matplotlib, well here's your answer! You can display the weights of your network as images to gain an understanding about how it works internally. For these to work well, try to use square numbers for the number of nodes in each layer of your network.
```
# This will show the weights for each node in the first hidden layer
nn.show()

# This will show the weights for each node in the second hidden layer
nn.show(2)
```
Here is what the first layer of an untrained network looks like:
![Untrained neural net](https://i.imgur.com/Boaa5mw.png)

You can also display an example of your training data like so:
```
# This will show the handwritten digit 5
img_show(training_data[0][0])
```
![MNIST dataset handwritten 5](https://i.imgur.com/KVqy097.png)

## The maths
A big part of machine learning and in particular neural networks is maths, specifically linear algebra and multi-variable calculus. Here I will try to explain the maths behind my code to demystify some of the more complicated parts. I do encourage you to watch the video series I linked to in the intro.

### Forward propagation
The easiest part is simply running the network, here I'll define some notation to make things easier.
The nth layers' weights, biases and activations are defined like this respectively:

![](https://latex.codecogs.com/svg.latex?w^{(n)}&space;=&space;\begin{bmatrix}&space;w_{0,0}^{(n)}&space;&&space;...&space;&&space;w_{0,i}^{(n)}&space;\\\\&space;:&&space;:&space;&&space;:&space;\\\\&space;w_{j,0}^{(n)}&&space;...&space;&&space;w_{j,i}^{(n)}&space;\end{bmatrix}) ![](https://latex.codecogs.com/svg.latex?b^{(n)}&space;=&space;\begin{bmatrix}&space;b_{0}^{(n)}&space;\\\\&space;:\\\\&space;b_{j}^{(n)}&space;\end{bmatrix})   ![](https://latex.codecogs.com/svg.latex?L^{(n)}&space;=&space;\begin{bmatrix}&space;L_{0}^{(n)}&space;\\\\&space;:\\\\&space;L_{j}^{(n)}&space;\end{bmatrix})

The operation of going from one layer to the next is where the linear algebra applies, and it's as follows:

![](https://latex.codecogs.com/svg.latex?L^{(n)}=\sigma(w^{(n)}\cdot&space;L^{(n-1)}&space;&plus;&space;b^{(n)}))

That funny symbol represents the sigmoid function which squishes all numbers on the number line between 0 and 1.

![](https://latex.codecogs.com/svg.latex?\sigma(x)=\frac{1}{1&plus;e^{-x}})

![](http://www.tikalon.com/blog/2011/sigmoid.gif)
