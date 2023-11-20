import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.python.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from datetime import datetime

def bgg_rating():
    # Ensure 'data' directory exists
    data_directory = 'data'
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"The '{data_directory}' directory does not exist.")

    # Get image filenames from the 'data' folder
    image_filenames = os.listdir(data_directory)
    image_paths = [os.path.join(data_directory, filename) for filename in image_filenames]

    # Create a DataFrame with filenames and corresponding ratings
    dataset = pd.DataFrame({'filename': image_filenames})
    # Assuming your dataset CSV also has 'avg_rating' column
    dataset['avg_rating'] = dataset['filename'].apply(lambda x: float(x.split('_rating_')[1].split('.jpg')[0]))
    
    # k-fold cross-validation
    k_folds = 5
    learning_rates = [0.001]  # Add more values to explore

    plt.figure(figsize=(12, 6))

    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        avg_train_loss = []
        avg_val_loss = []
        avg_train_mse = []
        avg_val_mse = []

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for train_index, val_index in kf.split(image_paths):
            train_data, train_labels = load_and_preprocess_images(dataset, np.array(image_paths)[train_index])
            val_data, val_labels = load_and_preprocess_images(dataset, np.array(image_paths)[val_index])

            model = create_model(learning_rate=lr)
            
            # Early Stopping Callback
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',  # Metric to monitor
                patience=3,           # Number of epochs with no improvement before training is stopped
                restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
            )

            # Train the model
            history = model.fit(
                train_data,
                train_labels,
                validation_data=(val_data, val_labels),
                epochs=10,
                verbose=0,
                callbacks=[ProgressCallback(), early_stopping]
            )

            # Track training and validation loss and mse values
            avg_train_loss.append(history.history['loss'])
            avg_val_loss.append(history.history['val_loss'])
            avg_train_mse.append(history.history['mse'])
            avg_val_mse.append(history.history['val_mse'])

        # Convert lists to numpy arrays and calculate the mean
        avg_train_loss = np.mean(np.array(avg_train_loss), axis=0)
        avg_val_loss = np.mean(np.array(avg_val_loss), axis=0)
        avg_train_mse = np.mean(np.array(avg_train_mse), axis=0)
        avg_val_mse = np.mean(np.array(avg_val_mse), axis=0)

        # Plot averaged learning curve
        plt.plot(avg_train_loss, label=f'Training Loss (LR={lr})')
        plt.plot(avg_val_loss, label=f'Validation Loss (LR={lr})')

        # Display averaged MSE values
        print(f"Avg Training MSE (LR={lr}): {avg_train_mse[-1]}")
        print(f"Avg Validation MSE (LR={lr}): {avg_val_mse[-1]}")

    plt.title('Averaged Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Function to load and preprocess images
def load_and_preprocess_images(dataset, image_paths, target_size=(224, 224)):
    images = []
    ratings = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img = img.resize(target_size)
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
        rating = dataset.loc[dataset['filename'] == os.path.basename(img_path), 'avg_rating'].values[0]
        ratings.append(rating)
    return np.array(images), np.array(ratings)

# Custom callback for progress messages
class ProgressCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.now()
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - Training:", end=' ')
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = datetime.now() - self.start_time
        print(f"Elapsed Time: {elapsed_time}, Loss: {logs['loss']}, Val Loss: {logs['val_loss']}, MSE: {logs['mse']}")

# Function to create the model
def create_model(learning_rate):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model

