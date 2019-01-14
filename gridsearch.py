import os, time, multiprocessing
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import offsetbox

from sklearn import svm, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV

import keras
from keras.datasets import mnist

import warnings
warnings.filterwarnings('ignore')

# Split data between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
train_images = X_train.astype('float32')
test_images = X_test.astype('float32')
train_images /= 255
test_images /= 255
train_labels = y_train
test_labels = y_test

# Grid search to find best classifier
# Generate a much smaller matrix with gammas and make matrix flat
gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
gamma_range = gamma_range.flatten()

# Generate matrix with all C and flatten matrix
C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
C_range = C_range.flatten()

parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}

svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=24, verbose=2)
grid_clsf.fit(train_images, train_labels)
sorted(grid_clsf.cv_results_.keys())

classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_
scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

plot_param_space_heatmap(scores, C_range, gamma_range)
plt.savefig('gridsearch.png')
