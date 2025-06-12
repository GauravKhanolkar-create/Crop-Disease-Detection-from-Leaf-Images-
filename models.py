# models.py

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import gc

def create_mini_model(num_classes, img_height, img_width):
    """Defines a very simple (mini) CNN architecture."""
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = layers.Conv2D(16, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lightweight_model(num_classes, img_height, img_width):
    """Defines a slightly more complex, but still lightweight, CNN architecture."""
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x) # Using GlobalAveragePooling2D reduces parameters significantly
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x) # Added dropout layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_ultra_fast_model(num_classes, img_height, img_width):
    """Defines a model using MobileNetV2 with transfer learning."""
    # Load MobileNetV2 with pre-trained ImageNet weights, excluding the top (classification) layer
    base_model = applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                          include_top=False,
                                          weights='imagenet') # 'imagenet' is the standard pre-trained weights
    
    # Freeze the base model layers
    base_model.trainable = False 
    
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False) # Important: set training=False when using base_model in inference mode
    x = layers.GlobalAveragePooling2D()(x) # Pooling layer to flatten feature maps
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Added dropout layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_gen, val_gen, epochs, callbacks=None):
    """Encapsulates the model training process."""
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1 # Show training progress
        )
        return history
    except Exception as e:
        print(f"Error training model: {e}")
        return None