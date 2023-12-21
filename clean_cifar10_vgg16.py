import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score




#!/usr/bin/env python3
"""
This script has the method
preprocess_data(X, Y):
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
        """ This method has the preprocess to train a model """
        X = X.astype('float32')
        X_p = K.applications.vgg16.preprocess_input(X)
        Y_p = K.utils.to_categorical(Y, 10)
        return(X_p, Y_p)

if __name__ == "__main__":
    (Xt, Yt), (X, Y) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(Xt, Yt)
    Xv_p, Yv_p = preprocess_data(X, Y)
    base_model = K.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg'
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
    callback += [K.callbacks.ModelCheckpoint('cifar10.h5',
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
print(f'VGG16 accuracy on CIFAR-10 Clean: {test_acc_poisoned * 100:.2f}%')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# Assuming you have the true labels (test_labels) and predicted labels (predicted_labels)
# Replace these with the actual variables you have in your code

# Evaluate the model on test data and get predictions
predictions = model.predict(Xv_p)
predicted_labels = np.argmax(predictions, axis=1)

# Create a confusion matrix
cm = confusion_matrix(Yv_p, predicted_labels)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()



# Assuming you have the true labels (test_labels) and predicted probabilities (predicted_probs)
# Replace these with the actual variables you have in your code

# Evaluate the model on test data and get predicted probabilities
predictions = model.predict(Xv_p)
predicted_probs = predictions
test_labels=Yv_p
# Binarize the labels if they are not binary
if len(np.unique(test_labels)) > 2:
    test_labels_bin = label_binarize(test_labels, classes=np.unique(test_labels))
    predicted_probs_bin = predicted_probs
else:
    test_labels_bin = test_labels
    predicted_probs_bin = predicted_probs[:, 1]  # Assuming you're working with binary classification

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(len(np.unique(test_labels))):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], predicted_probs_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_bin.ravel(), predicted_probs_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)

for i in range(len(np.unique(test_labels))):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.show()
