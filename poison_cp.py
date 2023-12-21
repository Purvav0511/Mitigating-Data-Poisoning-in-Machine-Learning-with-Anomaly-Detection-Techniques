#data loaded
import os
import numpy as np
import pickle
from PIL import Image
from keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images , test_images


poison_folder_path = "/Data/CIFAR-10/cp_poisons/num_poisons=25"  # Update with the actual path

# Iterate through folders
for folder_name in os.listdir(poison_folder_path):
    folder_path = os.path.join(poison_folder_path, folder_name)

    with open(os.path.join(folder_path, 'target.pickle'), 'rb') as handle:
        target_data = pickle.load(handle)

    with open(os.path.join(folder_path, 'poisons.pickle'), 'rb') as handle:
        poison_data = pickle.load(handle)

    with open(os.path.join(folder_path, 'base_indices.pickle'), 'rb') as handle:
        base_indices = pickle.load(handle)

    # Apply poison to CIFAR-10
        # Apply poison to CIFAR-10
    for base_index, (poison_img, poisoned_label) in zip(base_indices, poison_data):
        target_img, target_label = target_data
        train_images[base_index] = np.array(poison_img)/250
        train_labels[base_index] = np.array([poisoned_label])
        print(f"{base_index}, {poisoned_label}")

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
    X_p, Y_p = preprocess_data(train_images, train_labels)
    Xv_p, Yv_p = preprocess_data(test_images, test_labels)
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
print(f'VGG16 accuracy on Poison CIFAR-10 : {test_acc_poisoned * 100:.2f}%')