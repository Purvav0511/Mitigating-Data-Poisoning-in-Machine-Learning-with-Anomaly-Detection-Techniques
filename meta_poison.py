import pickle
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Load the meta-poisoned dataset
with open('/Data/CIFAR-10/Meta Poison/targetid_6/poisondataset-60.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract data from the dictionary
x_train_mp = data['xtrain']
y_train_mp = data['ytrain']
x_target = data['xtarget']
y_target = data['ytarget']
y_target_adv = data['ytargetadv']
x_valid = data['xvalid']
y_valid = data['yvalid']




#!/usr/bin/env python3
"""
This script has the method
preprocess_data(X, Y):
"""
import keras as K


def preprocess_data(X, Y):
        """ This method has the preprocess to train a model """
        X = X.astype('float32')
        X_p = K.applications.vgg16.preprocess_input(X)
        Y_p = K.utils.to_categorical(Y, 10)
        return(X_p, Y_p)

if __name__ == "__main__":
    X_p, Y_p = preprocess_data(x_train_mp, y_train_mp)
    Xv_p, Yv_p = preprocess_data(x_valid, y_valid)
    base_model = K.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg',
                                            classes=Y_p.shape[1]

                                            )

    model= K.Sequential()
    model.add(K.layers.UpSampling2D())
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []
    def decay(epoch):
        """ This method create the alpha"""
        return 0.001 / (1 + 1 * 30)
    callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
    callback += [K.callbacks.ModelCheckpoint('cifar10_vgg16_poison.h5',
                                             save_best_only=True,
                                             mode='min'
                                             )]
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=X_p, y=Y_p,
              batch_size=128,
              validation_data=(Xv_p, Yv_p),
              epochs=10, shuffle=True,
              callbacks=callback,
              verbose=1
              )

# Evaluate on clean test data
test_loss_poisoned, test_acc_poisoned = model.evaluate(Xv_p, Yv_p)
print(f'VGG16 accuracy on Meta Poison CIFAR-10 : {test_acc_poisoned * 100:.2f}%')


import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load the meta-poisoned dataset
with open('/content/drive/MyDrive/CIFAR-10/Meta Poison/targetid_6/poisondataset-60.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract data from the dictionary
x_train_mp = data['xtrain']
y_train_mp = data['ytrain']
x_target = data['xtarget']
y_target = data['ytarget']
y_target_adv = data['ytargetadv']
x_valid = data['xvalid']
y_valid = data['yvalid']




#!/usr/bin/env python3
"""
This script has the method
preprocess_data(X, Y):
"""
import keras as K


def preprocess_data(X, Y):
        """ This method has the preprocess to train a model """
        X = X.astype('float32')
        X_p = K.applications.vgg16.preprocess_input(X)
        Y_p = K.utils.to_categorical(Y, 10)
        return(X_p, Y_p)

if __name__ == "__main__":
    X_p, Y_p = preprocess_data(x_train_mp, y_train_mp)
    Xv_p, Yv_p = preprocess_data(x_valid, y_valid)
    base_model = K.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg',
                                            classes=Y_p.shape[1]

                                            )

    model= K.Sequential()
    model.add(K.layers.UpSampling2D())
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(512, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(256, activation=('relu')))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.Dense(10, activation=('softmax')))
    callback = []
    def decay(epoch):
        """ This method create the alpha"""
        return 0.001 / (1 + 1 * 30)
    callback += [K.callbacks.LearningRateScheduler(decay, verbose=1)]
    callback += [K.callbacks.ModelCheckpoint('cifar10_vgg16_poison.h5',
                                             save_best_only=True,
                                             mode='min'
                                             )]
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=X_p, y=Y_p,
              batch_size=128,
              validation_data=(Xv_p, Yv_p),
              epochs=10, shuffle=True,
              callbacks=callback,
              verbose=1
              )

# Evaluate on clean test data
test_loss_poisoned, test_acc_poisoned = model.evaluate(Xv_p, Yv_p)
print(f'VGG16 accuracy on Meta Poison CIFAR-10 : {test_acc_poisoned * 100:.2f}%')

## Isolation Forest MP

# Reshape images to 1D vectors
train_images_reshaped_mp = train_images.reshape((x_train_mp.shape[0], -1))

# Combine features and the poison column
#train_data_with_poison_column = np.column_stack((train_images_reshaped, poison_column))

# Apply Isolation Forest
isolation_forest = IsolationForest(n_estimators=100,contamination='auto', random_state=42) # contamination=0.1
isolation_forest.fit(train_images_reshaped_mp)

# Predict inliers (normal instances) and outliers (poison instances)
predicted_labels = isolation_forest.predict(train_images_reshaped_mp)

# Convert predictions to binary labels (1 for normal, -1 for outlier)
predicted_labels_binary = np.where(predicted_labels == 1, 0, 1)


## One Class SVM

# Fit One-Class SVM model
one_class_svm = OneClassSVM(nu=0.1)  # Adjust the hyperparameter 'nu' as needed
one_class_svm.fit(train_images_reshaped_mp)
# Predict anomalies on the test set


predicted_labels = one_class_svm.predict(train_images_reshaped_mp)

# Convert predictions to binary labels (1 for normal, -1 for outlier)
predicted_labels_binary = np.where(predicted_labels == 1, 0, 1)


accuracy = accuracy_score(poison_column, predicted_labels_binary)
print(f'One Class Svm Accuracy: {accuracy * 100:.2f}%')
