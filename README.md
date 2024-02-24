# Anomaly Detection for Federated Learning 
<br>
This repository contains code for anomaly detection using federated learning. The code is implemented in Python and utilizes PyTorch and TensorFlow/Keras for machine learning modeling.

## Introduction
Federated learning is a machine learning approach that enables training models across multiple decentralized devices or servers while keeping the data localized. This repository explores federated learning for anomaly detection in industrial Internet of Things (IIoT) environments.

## Dataset
The dataset used in this project is an IIoT dataset containing various features related to devices in an industrial setting. The dataset is preprocessed to handle missing values, categorical variables, and normalization.

## Code Structure
Main.ipynb: Jupyter Notebook containing the main code implementation.
IIoTDataset.csv: The dataset used for training and evaluation.
device_1.csv, device_2.csv, ..., device_5.csv: Split datasets for federated learning on different devices.
model_device_1.csv.h5, model_device_2.csv.h5, ..., model_device_5.csv.h5: Saved model weights for each device.
README.md: This file, providing an overview of the project.

## Usage
Data Preprocessing: The dataset is preprocessed to handle missing values, categorical variables, and normalization.
Model Training: LSTM models are trained using PyTorch and TensorFlow/Keras. Each device has its own model trained on its local data.
Federated Learning: Model weights from each device are aggregated to create a global model for anomaly detection.
Evaluation: The global model is evaluated on test data to assess its performance in anomaly detection.

## Requirements
Python 3.x
PyTorch
TensorFlow
Pandas
NumPy
Matplotlib
Seaborn
scikit-learn
