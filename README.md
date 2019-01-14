# MNIST-Analysis-Project

Machine Learning Assessment
---------------------------

A machine learning problem set to test the understanding of modern techniques at
the basic theoretical, intuitive as well as practical levels. With regard to
practical applications I use Python, as well related libraries such as Numpy,
Scikit and TensorFlow.

The project is centred around the MNIST data set of handwritten digits. It
consists of grayscale images of hand written digits, as well as the true
labels/digits of each image. The data is split in train and test sets. Many
problems relate to the classification task of predicting the true labels given 
the images.

Unsupervised Learning
---------------------

Performing a principle component analysis on the MNIST data set. Plots are
produced in the dimensionally reduced space using the principle components. I
also test the predictive accuracy of the PCA against boosted trees.

Support Vector Machines
-----------------------

Building a linear support vector classifier, then using grid search to find
optimal parameters for building a better non-linear classifier.

Neural Networks
---------------

Implementing from scratch, using a class, a feedforward neural network with 1
hidden layer of 100 hidden units merely using basic linear algebra computations
in Numpy. I also train the network using batch gradient descent and test your the implementation on MNIST.

Using TensorFlow, we implement a 2 layer neural network with 1200 and 1200
hidden units. This uses RELU activation function instead of sigmoids, as well as regularisation methods such as dropout to produce a model which has a test error
below 2%.

Data Science Challenge
----------------------

As a continuation of keras network, we improve the classifier by means of deeper
networks and convolution layers which are readily available as plug&play building
blocks.
