import os
import numpy as np
import pickle
from PIL import Image
from keras import datasets
import keras as K
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

poison_folder_path = "/content/drive/MyDrive/Data Poisoning/cp_poisons/num_poisons=25"  # Update with the actual path

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
    for base_index, (poison_img, poisoned_label) in zip(base_indices, poison_data):
        target_img, target_label = target_data
        train_images[base_index] = np.array(poison_img) / 250
        train_labels[base_index] = np.array([poisoned_label])
        print(f"{base_index}, {poisoned_label}")


def preprocess_data(X, Y):
    """ This method has the preprocess to train a model """
    X = X.astype('float32')
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return (X_p, Y_p)


if __name__ == "__main__":
    X_p, Y_p = preprocess_data(train_images, train_labels)
    Xv_p, Yv_p = preprocess_data(test_images, test_labels)
    base_model = K.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg',
                                            classes=Y_p.shape[1]

                                            )

    model = K.Sequential()
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

# Create a column to indicate poisoned instances
poison_column = np.zeros(train_images.shape[0])

# Iterate through folders
for folder_name in os.listdir(poison_folder_path):
    folder_path = os.path.join(poison_folder_path, folder_name)

    with open(os.path.join(folder_path, 'target.pickle'), 'rb') as handle:
        target_data = pickle.load(handle)

    with open(os.path.join(folder_path, 'poisons.pickle'), 'rb') as handle:
        poison_data = pickle.load(handle)

    with open(os.path.join(folder_path, 'base_indices.pickle'), 'rb') as handle:
        base_indices = pickle.load(handle)

    # Apply poison to CIFAR-10 and mark the poisoned instances in the column
    for base_index, (poison_img, poisoned_label) in zip(base_indices, poison_data):
        target_img, target_label = target_data
        train_images[base_index] = np.array(poison_img) / 255.0
        train_labels[base_index] = np.array([poisoned_label])
        poison_column[base_index] = 1

# Reshape images to 1D vectors
train_images_reshaped = train_images.reshape((train_images.shape[0], -1))

# Combine features and the poison column
train_data_with_poison_column = np.column_stack((train_images_reshaped, poison_column))

# Apply Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)  # contamination=0.1
isolation_forest.fit(train_images_reshaped)

# Predict inliers (normal instances) and outliers (poison instances)
predicted_labels = isolation_forest.predict(train_images_reshaped)

# Convert predictions to binary labels (1 for normal, -1 for outlier)
predicted_labels_binary = np.where(predicted_labels == 1, 0, 1)

# Evaluate the Isolation Forest performance
accuracy = accuracy_score(poison_column, predicted_labels_binary)
print(f'Isolation Forest Accuracy: {accuracy * 100:.2f}%')

# Fit One-Class SVM model
one_class_svm = OneClassSVM(nu=0.1)  # Adjust the hyperparameter 'nu' as needed
one_class_svm.fit(train_images_reshaped)

# Predict anomalies on the test set
predicted_labels = one_class_svm.predict(train_images_reshaped)

# Convert predictions to binary labels (1 for normal, -1 for outlier)
predicted_labels_binary = np.where(predicted_labels == 1, 0, 1)

accuracy = accuracy_score(poison_column, predicted_labels_binary)
print(f'One Class Svm Accuracy: {accuracy * 100:.2f}%')
