# Poison Detection Pipeline

## Overview

This project implements a Poison Detection Pipeline on the CIFAR-10 dataset. The pipeline includes multiple Python files, each applying different types of poison to the CIFAR-10 dataset. After poisoning the dataset, the files evaluate the VGG16 score and apply Isolation Forest and One-Class SVM for anomaly detection. The accuracy of these anomaly models is then calculated.

## File Descriptions

- **poison_cp.py**: Implements Convex Polytope (CP) poison on CIFAR-10 dataset and evaluates the VGG16 score and anomaly detection models.
- **poison_htbd.py**: Implements Hidden Trigger Backdoor Attack (HTBD) poison on CIFAR-10 dataset and evaluates the VGG16 score and anomaly detection models.
- **poison_fc.py**: Implements Feature Collision (FC) poison on CIFAR-10 dataset and evaluates the VGG16 score and anomaly detection models.
- **poison_mp.py**: Implements Meta Poison (MP) on CIFAR-10 dataset and evaluates the VGG16 score and anomaly detection models.


## Data Folder

The `data` folder contains the poisoned datasets used by the Python files. Ensure that the necessary data is present before running the scripts. The data is linked in this google drive below: 
https://drive.google.com/drive/folders/1rM0InYSEtoTcyuifzEpAliKHh3RCP0UT?usp=drive_link

## Dockerization

### Dockerfile

The Dockerfile in this repository helps create a Docker image for the Poison Detection Pipeline. It includes necessary dependencies, copies project files, and sets up the environment.

### Building Docker Image

To build the Docker image, navigate to the project directory and run:

```bash
docker build -t poison-detection-pipeline .

Running Docker Container
After building the Docker image, you can run a container:
```bash
docker run -it poison-detection-pipeline
