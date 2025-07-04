import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import io
import tf2onnx
import tempfile 

from config import IMG_HEIGHT, IMG_WIDTH

def setup_existing_dataset(path):
    if not path or not os.path.exists(path):
        return None
    return path

def setup_data_generators(data_dir, img_height, img_width, batch_size, augment=False):
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=0.2,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    try:
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            subset='training',
            class_mode='categorical',
            shuffle=True,
            seed=42,
        )
        val_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            subset='validation',
            class_mode='categorical',
            shuffle=False,
            seed=42,
        )
        return train_generator, val_generator
    except Exception as e:
        print(f"Error setting up data generators: {e}") 
        return None, None

def display_class_distribution(generator):
    try:
        class_indices = generator.class_indices
        class_names = list(class_indices.keys())
        
        class_counts = {name: 0 for name in class_names}
        for class_name, count in zip(class_names, generator.classes):
            class_counts[generator.class_indices_inverse[count]] += 1

        if not class_counts or sum(class_counts.values()) == 0:
            print("No class distribution data available. Generator might be empty.")
            return None

        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.title("Class Distribution")
        plt.ylabel("Number of Samples")
        plt.xlabel("Class Name")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error displaying class distribution: {e}")
        return None

def plot_training_history(history, title_suffix=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train Acc')
    ax[0].plot(history.history['val_accuracy'], label='Val Acc')
    ax[0].set_title('Accuracy ' + title_suffix)
    ax[0].legend()
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Loss ' + title_suffix)
    ax[1].legend()
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return fig

def plot_tsne(features, labels, class_names):
    try:
        tsne = TSNE(n_components=2, random_state=42)
        transformed = tsne.fit_transform(features)
        df = pd.DataFrame({'x': transformed[:,0], 'y': transformed[:,1], 'label': labels})
        fig, ax = plt.subplots(figsize=(8,6))
        
        unique_labels = np.unique(labels)
        for cl in unique_labels:
            subset = df[df['label']==cl]
            if 0 <= cl < len(class_names): 
                ax.scatter(subset['x'], subset['y'], label=class_names[int(cl)], alpha=0.7, s=50) 
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
        ax.set_title("t-SNE Clustering")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating t-SNE plot: {e}")
        return None

def plot_pca(features, labels, class_names):
    try:
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(features)
        df = pd.DataFrame({'x': transformed[:,0], 'y': transformed[:,1], 'label': labels})
        fig, ax = plt.subplots(figsize=(8,6))
        
        unique_labels = np.unique(labels)
        for cl in unique_labels:
            subset = df[df['label']==cl]
            if 0 <= cl < len(class_names): 
                ax.scatter(subset['x'], subset['y'], label=class_names[int(cl)], alpha=0.7, s=50)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("PCA Clustering")
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating PCA plot: {e}")
        return None

def preprocess_image(image_path, img_height, img_width):
    try:
        img = Image.open(image_path).convert('RGB').resize((img_width, img_height))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0) 
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_single_image(model, image_path, class_names):
    try:
        img_tensor = preprocess_image(image_path, IMG_HEIGHT, IMG_WIDTH)
        if img_tensor is None:
            return None, None, None
        
        preds = model.predict(img_tensor)[0]
        idx = np.argmax(preds)
        return class_names[idx], preds[idx], preds
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, None

def compute_gradcam(model, img_array, last_conv_layer_name="Conv_1"):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("No Conv2D layer found for Grad-CAM. Using a default name or skipping.")
            if "MobileNetV2" in model.name:
                last_conv_layer_name = 'Conv_1'
            else:
                print("Could not find a suitable convolutional layer for Grad-CAM.")
                return None


        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if len(predictions.shape) > 1:
                 pred_index = tf.argmax(predictions[0])
            else:
                 pred_index = tf.argmax(predictions)
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        if len(conv_outputs.shape) == 4:
            conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        return heatmap
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        return None

def overlay_gradcam(img_path, heatmap, alpha=0.4, colormap=plt.cm.jet):
    try:
        img = Image.open(img_path).convert('RGB')
        
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.LANCZOS)
        heatmap_np = np.array(heatmap_pil) / 255.0
        
        jet_colors = colormap(heatmap_np)[:,:,:3]
        jet_colors = np.uint8(255 * jet_colors)

        jet_pil = Image.fromarray(jet_colors)
        
        superimposed_img = Image.blend(img, jet_pil, alpha)
        return superimposed_img
    except Exception as e:
        print(f"Error overlaying Grad-CAM: {e}")
        return None

def export_to_onnx(model, onnx_path):
    try:
        spec = (tf.TensorSpec((None, IMG_HEIGHT, IMG_WIDTH, 3), tf.float32, name="input"),)
        
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        return True
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return False
