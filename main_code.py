import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

# Essential machine learning libraries for data handling and model evaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Image processing and system utilities
import cv2
import gc
import os
import warnings

# Deep learning frameworks for advanced neural network modeling
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Suppress warnings to keep output clean and focused
warnings.filterwarnings('ignore')

# Root path for the medical image dataset
# Note: Update this path to match your local directory structure
DATASET_PATH = r'path/to/your/lung_colon_image_set'

def load_and_explore_dataset(path):
    """
    Explore the dataset by visualizing sample images from each category.
    Helps in understanding the visual characteristics and diversity of the data.
    """
    # Retrieve image categories from directory structure
    classes = os.listdir(path)
    
    # Display a sample of images for each category
    for cat in classes:
        image_dir = os.path.join(path, cat)
        images = os.listdir(image_dir)
        
        # Create a visualization grid for the category
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sample Images for {cat} Category', fontsize=20)
        
        # Randomly select and display three images
        for i in range(3):
            # Pick a random image index
            k = np.random.randint(0, len(images))
            img = np.array(Image.open(os.path.join(image_dir, images[k])))
            ax[i].imshow(img)
            ax[i].axis('off')
        
        plt.tight_layout()
        plt.show()

def preprocess_images(path, img_size=256):
    """
    Prepare images for model training by standardizing size and format.
    Ensures consistent input for the neural network.
    """
    # Identify unique image categories
    classes = os.listdir(path)
    
    X, Y = [], []
    
    # Process images from each category
    for i, cat in enumerate(classes):
        # Find all JPEG images in the category folder
        images = glob(f"{path}/{cat}/*.jpeg")
        
        for image in images:
            # Read image using OpenCV
            img = cv2.imread(image)
            
            # Validate and process each image
            if img is not None:
                # Resize to a consistent dimension
                img = cv2.resize(img, (img_size, img_size))
                
                # Normalize pixel values to improve model performance
                # Scales pixel values between 0 and 1
                img = img / 255.0
                
                X.append(img)
                Y.append(i)
    
    # Convert lists to numpy arrays for model compatibility
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y, classes

def create_cnn_model(img_size, num_classes):
    """
    Construct a Convolutional Neural Network for medical image classification.
    Designed to extract meaningful features from complex medical images.
    """
    model = keras.models.Sequential([
        # First convolutional layer - initial feature extraction
        # Detects basic patterns and structures in the images
        layers.Conv2D(filters=32,
                      kernel_size=(5, 5),
                      activation='relu',
                      input_shape=(img_size, img_size, 3),
                      padding='same'),
        layers.MaxPooling2D(2, 2),

        # Second convolutional layer - more complex feature detection
        # Identifies more intricate patterns in the medical images
        layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same'),
        layers.MaxPooling2D(2, 2),

        # Third convolutional layer - advanced feature extraction
        # Captures sophisticated image characteristics
        layers.Conv2D(filters=128,
                      kernel_size=(3, 3),
                      activation='relu',
                      padding='same'),
        layers.MaxPooling2D(2, 2),

        # Flatten layer - prepare features for classification
        layers.Flatten(),
        
        # Fully connected layers for final classification
        # Combine extracted features to make a precise prediction
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        
        # Dropout layer to prevent overfitting
        # Helps improve model's generalization capability
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        # Output layer - final classification step
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model with appropriate optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, X_train, Y_train, X_val, Y_val, batch_size=64, epochs=10):
    """
    Train the neural network with built-in callbacks for optimization.
    Implements strategies to improve model performance and prevent overfitting.
    """
    # Early stopping callback to prevent unnecessary training
    # Stops when model performance plateaus
    early_stopping = keras.callbacks.EarlyStopping(
        patience=3,
        monitor='val_accuracy',
        restore_best_weights=True
    )

    # Learning rate reduction to fine-tune model training
    # Adjusts learning rate when performance stabilizes
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=2,
        factor=0.5,
        verbose=1
    )

    # Custom callback to halt training at high accuracy
    # Prevents unnecessary computational expense
    class AccuracyStoppingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') > 0.90:
                print('\nHigh accuracy reached, stopping training.')
                self.model.stop_training = True

    # Execute model training with carefully selected parameters
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping, learning_rate_reduction, AccuracyStoppingCallback()]
    )

    return history

def visualize_training_history(history):
    """
    Generate visualizations of model training performance.
    Helps in understanding how the model learned during training.
    """
    # Convert training history to DataFrame for easier plotting
    history_df = pd.DataFrame(history.history)
    
    # Create side-by-side plots for loss and accuracy
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    history_df[['loss', 'val_loss']].plot(ax=plt.gca())
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    history_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca())
    plt.title('Model Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, Y_val, classes):
    """
    Comprehensive model evaluation using multiple performance metrics.
    Provides insights into model's classification capabilities.
    """
    # Generate predictions on validation dataset
    Y_pred = model.predict(X_val)
    
    # Convert one-hot encoded labels to class indices
    Y_val_indices = np.argmax(Y_val, axis=1)
    Y_pred_indices = np.argmax(Y_pred, axis=1)
    
    # Generate confusion matrix to show prediction accuracy
    print("Confusion Matrix - Detailed Prediction Breakdown:")
    print(metrics.confusion_matrix(Y_val_indices, Y_pred_indices))
    
    # Produce comprehensive classification report
    print("\nDetailed Classification Report:")
    print(metrics.classification_report(Y_val_indices, Y_pred_indices, target_names=classes))

def main():
    """
    Main pipeline orchestrating the entire machine learning workflow.
    Coordinates data preprocessing, model training, and evaluation.
    """
    # Configuration parameters for consistent experimentation
    IMG_SIZE = 256
    TEST_SPLIT = 0.2
    EPOCHS = 10
    BATCH_SIZE = 64

    # Load and preprocess medical images
    X, Y, classes = preprocess_images(DATASET_PATH, img_size=IMG_SIZE)
    
    # One-hot encode labels for categorical classification
    one_hot_encoded_Y = pd.get_dummies(Y).values
    
    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, one_hot_encoded_Y,
        test_size=TEST_SPLIT,
        random_state=2022
    )
    
    # Display dataset information
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    
    # Create and train the neural network model
    model = create_cnn_model(IMG_SIZE, len(classes))
    history = train_model(model, X_train, Y_train, X_val, Y_val, 
                          batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    # Visualize training performance
    visualize_training_history(history)
    
    # Evaluate model's classification performance
    evaluate_model(model, X_val, Y_val, classes)

if __name__ == "__main__":
    # Configure GPU memory management to prevent allocation issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable dynamic memory allocation for GPU
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Fallback if GPU memory configuration fails
            print("Could not configure GPU memory growth")
    
    # Execute the main machine learning pipeline
    main()