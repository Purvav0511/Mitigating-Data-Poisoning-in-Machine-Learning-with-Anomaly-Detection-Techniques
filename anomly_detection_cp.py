import os
import numpy as np
import pickle
from PIL import Image
from keras import datasets
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

poison_folder_path = "/Data/CIFAR-10/cp_poisons/num_poisons=25"  # Update with the actual path

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
#train_data_with_poison_column = np.column_stack((train_images_reshaped, poison_column))

# Apply Isolation Forest
isolation_forest = IsolationForest(n_estimators=100,contamination='auto', random_state=42) # contamination=0.1
isolation_forest.fit(train_images_reshaped)

# Predict inliers (normal instances) and outliers (poison instances)
predicted_labels = isolation_forest.predict(train_images_reshaped)

# Convert predictions to binary labels (1 for normal, -1 for outlier)
predicted_labels_binary = np.where(predicted_labels == 1, 0, 1)

# Evaluate the Isolation Forest performance
#accuracy = accuracy_score(train_labels, predicted_labels_binary)
#print(f'Isolation Forest Accuracy: {accuracy * 100:.2f}%')


accuracy = accuracy_score(poison_column, predicted_labels_binary)
print(f'Isolation Forest Accuracy: {accuracy * 100:.2f}%')

from sklearn.svm import OneClassSVM

# Fit One-Class SVM model
one_class_svm = OneClassSVM(nu=0.1)  # Adjust the hyperparameter 'nu' as needed
one_class_svm.fit(train_images_reshaped)
# Predict anomalies on the test set


predicted_labels = one_class_svm.predict(train_images_reshaped)

# Convert predictions to binary labels (1 for normal, -1 for outlier)
predicted_labels_binary = np.where(predicted_labels == 1, 0, 1)


accuracy = accuracy_score(poison_column, predicted_labels_binary)
print(f'One Class Svm Accuracy: {accuracy * 100:.2f}%')
