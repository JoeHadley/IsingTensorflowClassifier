#Import packages
import numpy as np
import tensorflow as tf
from google.colab import files


#Upload data files
uploaded = files.upload()


#Load and preprocess data
def load_and_preprocess_data(data_path, labels_path, num_rows, side_length):
    with open(data_path, 'rb') as data_file:
        data = np.fromfile(data_file, dtype=np.int32)
    data = data.reshape(num_rows, side_length, side_length)

    with open(labels_path, 'rb') as label_file:
        labels = np.fromfile(label_file, dtype=np.int32)
    labels = labels.reshape(num_rows, 1)

    indices = np.random.permutation(num_rows)
    data = data[indices]
    labels = labels[indices]

    return data, labels

# Load and preprocess training and testing data
training_data, training_labels = load_and_preprocess_data('trainingData.dat', 'trainingLabels.dat', 1000, 10)
testing_data, testing_labels = load_and_preprocess_data('testingData.dat', 'testingLabels.dat', 200, 10)


# Create the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(10, 10)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


# Train the model
model.fit(training_data, training_labels, epochs=5, validation_data=(testing_data, testing_labels))