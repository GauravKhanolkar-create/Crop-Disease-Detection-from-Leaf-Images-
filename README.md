## Crop Disease Detection from Leaf Images

## 🌿 Project Overview

This project implements a full pipeline for detecting crop diseases from leaf images using Convolutional Neural Networks (CNNs) and Transfer Learning. It features a Streamlit web application for interactive predictions, model training with cross-validation, performance analysis, and model export for deployment.

The goal is to provide a robust solution for identifying plant diseases early, helping farmers and agricultural enthusiasts take timely action.

## ✨ Features

Image Classification: Classifies crop diseases from leaf images using deep learning models.

Data Preprocessing: Handles image resizing and data normalization.

Image Augmentation: Applies various augmentation techniques (rotation, flip, zoom) to enhance model generalization.

Multiple Model Architectures: Includes a basic CNN, a lightweight CNN, and a MobileNetV2-based model (leveraging transfer learning).

Model Training: Supports training with a single train/validation split.

K-Fold Cross-Validation: Provides an option for K-Fold cross-validation for more robust model evaluation.

Training History Visualization: Plots training and validation accuracy and loss curves.

Model Comparison Dashboard: Summarizes performance (accuracy, loss, parameters) across different trained models.

Real-time Prediction Web App: A Streamlit interface to upload a leaf image and get instant disease predictions.

Model Interpretability (Grad-CAM): Visualizes regions of interest for MobileNetV2 predictions using Grad-CAM heatmaps.

Clustering Visualization: Uses t-SNE and PCA on image embeddings to visualize how different diseases cluster in the feature space.

ONNX Model Export: Allows exporting trained models to ONNX format for efficient deployment.

## 🚀 Technologies Used
Python 3.x

TensorFlow / Keras: For building and training CNN models.

Streamlit: For creating the interactive web application.

NumPy: For numerical operations.

Pandas: For data handling and analysis.

Matplotlib & Seaborn: For data visualization (training plots, class distribution, clustering).

Scikit-learn: For data splitting and K-Fold cross-validation.

Pillow (PIL): For image manipulation.

tf2onnx: For converting TensorFlow/Keras models to ONNX format.

## 💾 Dataset
This project is designed to work with image datasets of crop diseases, such as the PlantVillage dataset.

## 🚀 Usage
Prepare your dataset: Download the PlantVillage dataset (or similar) and place it in a folder. Update the data_dir_input in app.py or directly in the Streamlit sidebar to the path of your dataset folder (e.g., plantvillage_dataset).

Run the Streamlit application:

streamlit run app.py

This will open the application in your default web browser.

Interact with the App:

Sidebar: Adjust Batch Size, Epochs, and select the Model Architecture.

Dataset Overview: View basic info and class distribution.

Train Models: Click "Start Training" to begin the model training process (with or without K-Fold Cross Validation).

Model Comparison Dashboard: See a summary of trained models' performance.

Instant Disease Detection: Upload an image to get a prediction.

Clustering & Visualization: Generate t-SNE and PCA plots to understand feature space clustering.

Export Trained Model to ONNX: Convert and download your best model for deployment.

