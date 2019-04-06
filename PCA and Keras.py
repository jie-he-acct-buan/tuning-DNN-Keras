import numpy as np
import pandas as pd
import sklearn
import mglearn
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', None)

np.random.seed(13)

##############################################################################
from keras.datasets import mnist
from keras.utils import np_utils
(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10 # class size
input_unit_size = 28*28 # input vector size

X_train = X_train.reshape(X_train.shape[0], input_unit_size)
X_test  = X_test.reshape(X_test.shape[0], input_unit_size)
X_train_image = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols) 
X_test_image = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

###############################################################################
# Apply K-means, with and without PCA, to do clusters.

def draw_digit(data, row, col, n):
    plt.subplot(row, col, n)    
    plt.imshow(data)
    plt.gray()

image_shape = X_train_image_sample[0].shape
###############################################################################
# K-Means based on raw features
kmeans = sklearn.cluster.KMeans(n_clusters = num_classes)
kmeans.fit(X_train)
X_train_labels = kmeans.labels_

(XX_train, yy_train), (XX_test, yy_test) = mnist.load_data()

for i in range(10):
    print('cluster with label ', i)
    j = 0
    for XX, label in zip(XX_train[:1000], X_train_labels):
        if label == i:
            j += 1
            if j <=20:
                draw_digit(XX, 4, 5, j)
            else:
                break
    plt.figure(figsize=(20, 20))
    plt.show()
    print('\n\n')
  
###############################################################################
# K-Means based on PCA features 
pca_try = sklearn.decomposition.PCA()
pca_try.fit(X_train)
cumsum = np.cumsum(pca_try.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print('number of pca dimensions accounting for more than 95% variance: \n', d)

pca = sklearn.decomposition.PCA(n_components = 154)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)

kmeans_pca = sklearn.cluster.KMeans(n_clusters = num_classes)
kmeans_pca.fit(X_train_pca)

X_train_pca_labels = kmeans_pca.labels_

(XX_train, yy_train), (XX_test, yy_test) = mnist.load_data()

for i in range(10):
    print('cluster with label ', i)
    j = 0
    for XX, label in zip(XX_train[:1000], X_train_pca_labels):
        if label == i:
            j += 1
            if j <=20:
                draw_digit(XX, 4, 5, j)
            else:
                break
    plt.show()
    print('\n\n')


###############################################################################
'''
Train a two-layer neural network:
    Hidden layer: number of hidden units = 128, activation = ReLu
    Use Dropout Strategy
    Output layer: activation = softmax
    loss function: categorical_crossentropy
    optimizer: SGD
    metrics: accuracy

    
Uuse grid search to tune the dropout rate and the learning rate in SGD optimizer.

Use KerasClassifier 
        (https://keras.io/scikit-learn-api/)

References: https://chrisalbon.com/deep_learning/keras/tuning_neural_network_hyperparameters/ 

'''


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(dropout_rate = 0.1, learning_rate = 0.1):
    model = Sequential()
    model.add(Dense(128, input_dim=input_unit_size, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn = create_model)

params = {'dropout_rate': [0, 0.1, 0.2, 0.3, 0.4, 0.5], 'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5]}

grid = GridSearchCV(estimator = model, param_grid = params)
grid_result = grid.fit(X_train, y_train)

print('The best dropout_rate and the best learning_rate: ', grid_result.best_params_, '\n\n\n')
print('The best score: ', grid_result.best_score_, '\n\n\n')

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("mean: %f, std: %f, with: %r" % (mean, stdev, param))


###############################################################################
'''
Explore the correlation between model preformance and the number of units

Plot a graph where 
    the x-axis is the number of hidden units, 
    and the y-axis is the model performance (two lines: one with test loss, and one with test accuracy)
    
'''
from keras.callbacks import History
history = History()
    

units = [2**i for i in range(1, 15)]
loss = []
accuracy = []
for i in units:
    print('\n', 'number of hidden units: ', i)
    model = Sequential()
    model.add(Dense(i, input_dim=input_unit_size, activation='relu'))
    model.add(Dropout(0))
    model.add(Dense(num_classes, activation='softmax'))
    sgd = SGD(lr = 0.5)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
    model_result = model.fit(X_train, y_train, validation_split=0.2)
    loss.append(model_result.history['val_loss'])
    accuracy.append(model_result.history['val_acc'])
    
print('\n\n\n', 'loss: ', '\n', loss)
print('\n\n\n', 'accuracy: ', '\n', accuracy)
print('\n\n\n')

plt.plot(units, loss, marker='o', color = 'r', label = 'loss')
plt.plot(units, accuracy, marker='^', color = 'b', label = 'accuracy')
plt.title('correlation between model preformance and the number of units')
plt.xscale('log')
plt.xlabel('number of hidden units')
plt.ylabel('loss / accuracy')
plt.xticks(units, units, rotation = 45)
plt.legend(loc = 1)
 
###############################################################################



