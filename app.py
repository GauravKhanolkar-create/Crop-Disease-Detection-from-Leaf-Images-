# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import os
import tempfile # For handling uploaded files

# Import modules
from config import IMG_HEIGHT, IMG_WIDTH
from utils import (
    setup_existing_dataset, setup_data_generators, display_class_distribution,
    plot_training_history, plot_confusion_matrix, plot_tsne, plot_pca,
    preprocess_image, predict_single_image, compute_gradcam, overlay_gradcam,
    export_to_onnx
)
from models import create_mini_model, create_lightweight_model, create_ultra_fast_model, train_model

def main():
    st.title("🌾 Crop Disease Detection - Full Pipeline with Advanced Features")
    
    # Sidebar inputs
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
    epochs = st.sidebar.slider("Epochs", 3, 20, 8) # Reduced max epochs for quicker demo
    data_dir_input = st.sidebar.text_input("Dataset Folder Path", value="D:\Final Project\Crop Disease Detection from Leaf Images\PlantVillage") # CHANGE THIS
    model_type = st.sidebar.selectbox(
        "Select Model Architecture",
        ["Mini Model (Fastest)", "Lightweight CNN (Fast)", "MobileNet-Fast (Balanced)"]
    )
    
    # Speed mode description - currently just informational in UI
    st.sidebar.radio(
        "Speed vs Accuracy Priority",
        ["Ultra Fast (3-5 min)", "Fast (5-10 min)", "Balanced (10-15 min)"],
        index=2
    )
    
    st.sidebar.markdown("---")
    
    # Initialize session state
    if 'model_histories' not in st.session_state:
        st.session_state.model_histories = {}
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'trained_models_metrics' not in st.session_state:
        st.session_state.trained_models_metrics = {}
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    if 'train_gen' not in st.session_state: # Store generator for later use
        st.session_state.train_gen = None
    if 'val_gen' not in st.session_state:
        st.session_state.val_gen = None

    data_dir = setup_existing_dataset(data_dir_input)
    if data_dir is None:
        st.error("❌ Dataset not found. Please check the path and try again.")
        st.stop() # Stop execution if dataset path is invalid
    
    # Dataset Info and Quick Analysis
    st.header("📊 Dataset Overview & Analysis")
    
    # Only setup generators once or if path changes
    if st.session_state.train_gen is None or st.session_state.train_gen.directory != data_dir:
        train_gen, val_gen = setup_data_generators(data_dir, IMG_HEIGHT, IMG_WIDTH, batch_size, augment=True)
        if train_gen is None:
            st.error("Failed to load dataset. Please check the dataset structure (e.g., classes as subfolders).")
            st.stop()
        st.session_state.train_gen = train_gen
        st.session_state.val_gen = val_gen
    else:
        train_gen = st.session_state.train_gen
        val_gen = st.session_state.val_gen
        train_gen.batch_size = batch_size # Update batch size if it changed in sidebar
        val_gen.batch_size = batch_size


    st.success(f"Dataset loaded with {len(train_gen.class_indices)} classes.")
    st.write(f"Training samples: {train_gen.samples}")
    st.write(f"Validation samples: {val_gen.samples}")
    
    with st.expander("Show Class Distribution"):
        fig = display_class_distribution(train_gen)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Could not display class distribution. Ensure dataset is correctly structured.")
    
    # K-Fold Cross Validation Section
    st.header("🔁 K-Fold Cross Validation (Optional)")
    kfold_enabled = st.checkbox("Enable K-Fold Cross Validation", value=False)
    kfold_splits = st.number_input("Number of folds", min_value=2, max_value=10, value=5, step=1, disabled=not kfold_enabled)
    
    def kfold_train_and_evaluate(create_model_func, data_dir_path, img_height, img_width, batch_size_kfold, kfold_splits_num):
        try:
            all_histories = []
            all_metrics = []

            # Prepare dataframe with image paths and labels for K-Fold
            image_paths = []
            labels = []
            
            # Get class indices from the main generator
            class_to_idx = st.session_state.train_gen.class_indices
            idx_to_class = {v: k for k, v in class_to_idx.items()}

            for class_name, class_idx in class_to_idx.items():
                folder = os.path.join(data_dir_path, class_name)
                if os.path.exists(folder):
                    for fname in os.listdir(folder):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_paths.append(os.path.join(folder, fname))
                            labels.append(class_idx)
            
            if not image_paths:
                st.error("No images found in the dataset for K-Fold. Please check the path and structure.")
                return None, None, None

            df = pd.DataFrame({
                'filename': image_paths,
                'class': [idx_to_class[l] for l in labels]
            })

            kf = KFold(n_splits=kfold_splits_num, shuffle=True, random_state=42)
            
            # Use a non-augmenting datagen for K-Fold splits, augmentation handled in main train_model
            kfold_datagen = ImageDataGenerator(rescale=1./255) 

            for fold_num, (train_idx, val_idx) in enumerate(kf.split(df['filename']), 1):
                st.write(f"Training fold {fold_num}/{kfold_splits_num}...")

                df_train = df.iloc[train_idx]
                df_val = df.iloc[val_idx]
                
                # Check if train/val splits are not empty
                if df_train.empty or df_val.empty:
                    st.warning(f"Fold {fold_num}: Empty training or validation split. Skipping fold.")
                    continue

                train_gen_fold = kfold_datagen.flow_from_dataframe(
                    df_train,
                    x_col='filename',
                    y_col='class',
                    target_size=(img_height, img_width),
                    batch_size=batch_size_kfold,
                    class_mode='categorical',
                    shuffle=True,
                    seed=42
                )

                val_gen_fold = kfold_datagen.flow_from_dataframe(
                    df_val,
                    x_col='filename',
                    y_col='class',
                    target_size=(img_height, img_width),
                    batch_size=batch_size_kfold,
                    class_mode='categorical',
                    shuffle=False,
                    seed=42
                )

                # Create a fresh model for each fold to avoid weight leakage
                model = create_model_func(len(class_to_idx), img_height, img_width)
                history = train_model(model, train_gen_fold, val_gen_fold, epochs)
                
                if history: # Only proceed if training was successful
                    val_loss, val_acc = model.evaluate(val_gen_fold, verbose=0)
                    all_histories.append(history)
                    all_metrics.append({
                        'fold': fold_num,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'model': model
                    })
                else:
                    st.warning(f"Fold {fold_num} training failed. Skipping.")
                
                del model # Clear model from memory
                tf.keras.backend.clear_session()
                gc.collect()

            return all_histories, all_metrics, idx_to_class
        except Exception as e:
            st.error(f"Error in K-fold training: {e}")
            return None, None, None

    
    # Train Button and Training Logic
    st.header("🚀 Train Models")
    
    if st.button("Start Training (with K-Fold if enabled)"):
        with st.spinner("Training started... This may take a while depending on your dataset and epochs."):
            try:
                # Determine which create_model function to use
                if model_type == "Mini Model (Fastest)":
                    create_model_func = create_mini_model
                elif model_type == "Lightweight CNN (Fast)":
                    create_model_func = create_lightweight_model
                else:
                    create_model_func = create_ultra_fast_model
                
                if kfold_enabled:
                    # K-Fold will handle its own data splitting and generator creation
                    result = kfold_train_and_evaluate(create_model_func, data_dir, IMG_HEIGHT, IMG_WIDTH, batch_size, kfold_splits)
                    if result and result[0] is not None:
                        histories, metrics, class_map = result
                        st.success(f"K-Fold training completed on {kfold_splits} folds.")
                        st.session_state.class_names = list(class_map.values())
                        st.session_state.model_histories[model_type] = histories
                        st.session_state.trained_models_metrics[model_type] = metrics
                        # Store the model from the last fold for single image inference
                        st.session_state.models[model_type] = metrics[-1]['model'] 
                    else:
                        st.error("K-Fold training failed or produced no results.")
                else:
                    # Single train-val split training (uses the initial train_gen, val_gen with augmentation)
                    st.info(f"Training {model_type} with single train-validation split...")
                    model = create_model_func(len(train_gen.class_indices), IMG_HEIGHT, IMG_WIDTH)
                    history = train_model(model, train_gen, val_gen, epochs)
                    if history is not None:
                        st.success("Training completed!")
                        st.session_state.class_names = list(train_gen.class_indices.keys())
                        st.session_state.model_histories[model_type] = [history] # Store as a list for consistency
                        st.session_state.trained_models_metrics[model_type] = [{'val_loss': history.history['val_loss'][-1], 'val_accuracy': history.history['val_accuracy'][-1], 'model': model}]
                        st.session_state.models[model_type] = model
                    else:
                        st.error("Training failed.")
                
                gc.collect()
                tf.keras.backend.clear_session()
            except Exception as e:
                st.error(f"An unexpected error occurred during training: {e}")
    
    # Show training results and charts for selected model
    if model_type in st.session_state.model_histories:
        st.header(f"📈 Training Results for {model_type}")
        histories = st.session_state.model_histories[model_type]
        metrics = st.session_state.trained_models_metrics[model_type]
        
        if kfold_enabled and histories:
            accs = [m['val_accuracy'] for m in metrics if 'val_accuracy' in m]
            losses = [m['val_loss'] for m in metrics if 'val_loss' in m]
            
            if accs: # Ensure there are results from folds
                st.write(f"Average Validation Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
                st.write(f"Average Validation Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
                
                # Plot all folds accuracy curves
                fig, ax = plt.subplots(figsize=(8,5))
                for i, hist in enumerate(histories):
                    ax.plot(hist.history.get('val_accuracy', []), label=f'Fold {i+1}')
                ax.set_title(f'Validation Accuracy per Fold for {model_type}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("No valid fold results to display.")
        elif histories: # Single train-val split
            fig = plot_training_history(histories[0])
            st.pyplot(fig)
        else:
            st.info("No training history available for the selected model type.")
            
    # Model Comparison Dashboard
    st.header("🧮 Model Comparison Dashboard")
    if st.session_state.trained_models_metrics:
        comparison_data = []
        for mtype, metrics_list in st.session_state.trained_models_metrics.items():
            val_accs = [m['val_accuracy'] for m in metrics_list if 'val_accuracy' in m]
            avg_acc = np.mean(val_accs) if val_accs else 0
            val_losses = [m['val_loss'] for m in metrics_list if 'val_loss' in m]
            avg_loss = np.mean(val_losses) if val_losses else 0
            
            model_params = 0
            if mtype in st.session_state.models and st.session_state.models[mtype] is not None:
                model_params = st.session_state.models[mtype].count_params()

            comparison_data.append({
                'Model': mtype,
                'Avg Val Accuracy': avg_acc,
                'Avg Val Loss': avg_loss,
                'Parameters': model_params,
            })
        df_compare = pd.DataFrame(comparison_data)
        df_compare = df_compare.sort_values(by='Avg Val Accuracy', ascending=False)
        st.dataframe(df_compare.style.format({"Avg Val Accuracy": "{:.2%}", "Avg Val Loss": "{:.4f}", "Parameters": "{:,}"}))
    else:
        st.info("Train models first to see comparison.")
    
    # Instant Prediction Section
    st.header("🔍 Instant Disease Detection")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        if model_type not in st.session_state.models or st.session_state.models[model_type] is None:
            st.warning("Please train the selected model first to enable prediction.")
        else:
            st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)
            model = st.session_state.models[model_type]
            class_names = st.session_state.class_names
            
            # Save uploaded file temporarily to disk to read with PIL (needed for Grad-CAM)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filepath = temp_file.name
            
            result = predict_single_image(model, temp_filepath, class_names)
            if result[0] is not None:
                pred_class, pred_prob, pred_probs_all = result
                st.markdown(f"### Prediction: **{pred_class}** with confidence {pred_prob:.2%}")
                
                # Show probabilities for all classes
                prob_df = pd.DataFrame({
                    "Class": class_names,
                    "Probability": pred_probs_all
                }).sort_values(by="Probability", ascending=False)
                st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
                
                # Grad-CAM visualization for MobileNet and similar architectures
                if model_type == "MobileNet-Fast (Balanced)": # Only attempt Grad-CAM for MobileNet
                    st.subheader("Grad-CAM Visualization")
                    img_array_for_gradcam = preprocess_image(temp_filepath, IMG_HEIGHT, IMG_WIDTH)
                    if img_array_for_gradcam is not None:
                        # Attempt to find the last conv layer for MobileNetV2
                        # The actual name can vary based on the specific Keras implementation
                        # Common names: 'Conv_1', 'block_16_project_BN' (for the block before classifier)
                        # Let's try 'Conv_1' as a common last conv layer before final average pooling
                        heatmap = compute_gradcam(model, img_array_for_gradcam, last_conv_layer_name='Conv_1') 
                        if heatmap is not None:
                            superimposed_img = overlay_gradcam(temp_filepath, heatmap)
                            if superimposed_img is not None:
                                st.image(superimposed_img, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
                            else:
                                st.warning("Could not overlay Grad-CAM heatmap.")
                        else:
                            st.warning("Could not compute Grad-CAM heatmap. Check model architecture and layer names.")
                    else:
                        st.warning("Could not preprocess image for Grad-CAM.")
            
            # Clean up temporary file
            try:
                os.unlink(temp_filepath)
            except Exception as e:
                st.warning(f"Could not delete temporary file: {e}")
    
    # Export model to ONNX
    st.header("💾 Export Trained Model to ONNX")
    if st.session_state.models:
        model_options = {name: model_obj for name, model_obj in st.session_state.models.items() if model_obj is not None}
        if model_options:
            selected_model_name = st.selectbox("Select Model to Export", list(model_options.keys()))
            selected_model_obj = model_options[selected_model_name]
            onnx_filename = st.text_input("ONNX Filename", value=f"{selected_model_name.replace(' ', '_').lower()}.onnx")
            if st.button("Export Model to ONNX"):
                with st.spinner(f"Exporting {selected_model_name} to ONNX..."):
                    if export_to_onnx(selected_model_obj, onnx_filename):
                        st.success(f"Model exported successfully to {onnx_filename}")
                        st.download_button(
                            label=f"Download {onnx_filename}",
                            data=open(onnx_filename, "rb").read(),
                            file_name=onnx_filename,
                            mime="application/octet-stream"
                        )
                    else:
                        st.error(f"Failed to export {selected_model_name} to ONNX.")
        else:
            st.info("No trained models available for export.")
    else:
        st.info("Train a model first to export.")

    # Clustering & Visualization Section
    st.header("🔬 Clustering & Visualization")
    if st.session_state.models and val_gen and st.session_state.class_names:
        current_model_obj = st.session_state.models.get(model_type)
        if current_model_obj is not None:
            if st.button("Generate Clustering Plots"):
                try:
                    with st.spinner("Generating embeddings and clustering plots (this may take a moment)..."):
                        # Ensure val_gen is reset and ready for use
                        val_gen.reset() 
                        
                        # Extract embeddings from penultimate layer for validation data
                        # This works for models with a final Dense layer for classification
                        # The layer before the final dense is usually the feature embedding layer
                        if len(current_model_obj.layers) < 2:
                            st.warning("Model has too few layers for embedding extraction.")
                            embedding_model = None
                        else:
                            embedding_model = tf.keras.Model(inputs=current_model_obj.input, outputs=current_model_obj.layers[-2].output)
                        
                        if embedding_model:
                            val_features = []
                            val_labels = []
                            
                            # Process validation data in batches
                            # Limit number of batches to prevent excessive computation for large datasets
                            num_batches_to_process = min(len(val_gen), 20) # Process max 20 batches
                            
                            for i in range(num_batches_to_process):
                                batch_data, batch_labels = val_gen.next() # Use .next() for explicit batch retrieval
                                batch_features = embedding_model.predict(batch_data, verbose=0)
                                val_features.append(batch_features)
                                val_labels.extend(np.argmax(batch_labels, axis=1))
                                
                                # Clear batch data to free memory
                                del batch_data, batch_labels, batch_features
                                gc.collect()
                            
                            if val_features:
                                # Combine all features and labels
                                embeddings = np.vstack(val_features)
                                labels = np.array(val_labels)
                                
                                st.subheader("t-SNE Visualization")
                                fig_tsne = plot_tsne(embeddings, labels, st.session_state.class_names)
                                if fig_tsne:
                                    st.pyplot(fig_tsne)
                                else:
                                    st.warning("t-SNE plot could not be generated.")
                                
                                st.subheader("PCA Visualization")
                                fig_pca = plot_pca(embeddings, labels, st.session_state.class_names)
                                if fig_pca:
                                    st.pyplot(fig_pca)
                                else:
                                    st.warning("PCA plot could not be generated.")
                            else:
                                st.warning("No embeddings could be extracted. Check validation generator data.")
                        else:
                            st.warning("Could not create embedding model. Make sure the selected model has enough layers.")
                            
                except Exception as e:
                    st.error(f"Error generating clustering plots: {e}")
        else:
            st.info("The currently selected model has not been trained yet. Please train it first.")
    else:
        st.info("Train a model first and ensure validation data is loaded to visualize clustering.")


if __name__ == "__main__":
    main()