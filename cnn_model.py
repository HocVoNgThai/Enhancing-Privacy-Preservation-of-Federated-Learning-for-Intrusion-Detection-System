import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Load dataset from CSV
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    features = data.iloc[:, :-1].values  
    labels = pd.get_dummies(data.iloc[:, -1]).values  
    return features, labels

# CNN model definition
def create_cnn_model(input_shape, output_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(csv_file, batch_size=512, epochs=10):
    x_train, y_train = load_data(csv_file)
    input_shape = (x_train.shape[1], 1)
    output_shape = y_train.shape[1]

    # Reshape input for Conv1D
    x_train = np.expand_dims(x_train, axis=-1)
    
    model = create_cnn_model(input_shape, output_shape)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save('trained_cnn_model.h5')
    print("Model training complete. Model saved as 'trained_cnn_model.h5'.")

# Example usage
csv_file = 'your_dataset.csv'  # Change this to your CSV file
train_model(csv_file)
