#Import packages
import numpy as np
import tensorflow as tf
from google.colab import files
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

#Load and preprocess data
def load_and_preprocess_data(data_path, labels_path, temps_path, num_rows, side_length):
    with open(data_path, 'rb') as data_file:
        data = np.fromfile(data_file, dtype=np.int32)
    data = data.reshape(num_rows, side_length, side_length)

    with open(labels_path, 'rb') as label_file:
        labels = np.fromfile(label_file, dtype=np.int32)
    labels = labels.reshape(num_rows, 1)

    with open(labels_path, 'rb') as temps_file:
        temps = np.fromfile(temps_file, dtype=np.float64)
    temps = temps.reshape(num_rows, 1)

    indices = np.random.permutation(num_rows)
    data = data[indices]
    labels = labels[indices]
    temps = temps[indices]

    return data, labels, temps

# Load and preprocess training and testing data

training_data_number = 2500

#training_data, training_labels = load_and_preprocess_data('TrainingData.dat', 'TrainingLabels.dat', 1000, 10)
#testing_data, testing_labels = load_and_preprocess_data('TestingData.dat', 'TestingLabels.dat', 300, 10)
training_data, training_labels, training_temps = load_and_preprocess_data('TrainingData.dat', 'TrainingLabels.dat', 'TrainingTemps.dat', 2500, 10)

# Define the neural network model
def create_model():
    model = tf.keras.models.Sequential([
        Flatten(input_shape=(10, 10)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(2)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model




early_stopping = EarlyStopping(
  monitor='val_loss',     # Monitor validation loss
  patience=3,             # Stop after 3 epochs with no improvement
  restore_best_weights=True  # Restore the best model weights
)


# We will train 5 models on different parts of the data.

size = training_data_number // 5  # Use integer division to ensure an integer size

tempChunk1 = training_labels[0:size - 1]
tempChunk2 = training_labels[size:2 * size - 1]
tempChunk3 = training_labels[2 * size:3 * size - 1]
tempChunk4 = training_labels[3 * size:4 * size - 1]
tempChunk5 = training_labels[4 * size:5 * size - 1]
NN1TrainingTemps = np.concatenate((tempChunk1,tempChunk2,tempChunk3,tempChunk4),axis = 0)
NN1TestingTemps = tempChunk5
NN2TrainingTemps = np.concatenate((tempChunk1,tempChunk2,tempChunk3,tempChunk5),axis = 0)
NN2TestingTemps = tempChunk4
NN3TrainingTemps = np.concatenate((tempChunk1,tempChunk2,tempChunk4,tempChunk5),axis = 0)
NN3TestingTemps = tempChunk3
NN4TrainingTemps = np.concatenate((tempChunk1,tempChunk3,tempChunk4,tempChunk5),axis = 0)
NN4TestingTemps = tempChunk2
NN5TrainingTemps = np.concatenate((tempChunk2,tempChunk3,tempChunk4,tempChunk5),axis = 0)
NN5TestingTemps = tempChunk1



labelChunk1 = training_labels[0:size - 1]
labelChunk2 = training_labels[size:2 * size - 1]
labelChunk3 = training_labels[2 * size:3 * size - 1]
labelChunk4 = training_labels[3 * size:4 * size - 1]
labelChunk5 = training_labels[4 * size:5 * size - 1]
NN1TrainingLabels = np.concatenate((labelChunk1,labelChunk2,labelChunk3,labelChunk4),axis = 0)
NN1TestingLabels = labelChunk5
NN2TrainingLabels = np.concatenate((labelChunk1,labelChunk2,labelChunk3,labelChunk5),axis = 0)
NN2TestingLabels = labelChunk4
NN3TrainingLabels = np.concatenate((labelChunk1,labelChunk2,labelChunk4,labelChunk5),axis = 0)
NN3TestingLabels = labelChunk3
NN4TrainingLabels = np.concatenate((labelChunk1,labelChunk3,labelChunk4,labelChunk5),axis = 0)
NN4TestingLabels = labelChunk2
NN5TrainingLabels = np.concatenate((labelChunk2,labelChunk3,labelChunk4,labelChunk5),axis = 0)
NN5TestingLabels = labelChunk1


dataChunk1 = training_data[0:size - 1]
dataChunk2 = training_data[size:2 * size - 1]
dataChunk3 = training_data[2 * size:3 * size - 1]
dataChunk4 = training_data[3 * size:4 * size - 1]
dataChunk5 = training_data[4 * size:5 * size - 1]

NN1TrainingData = np.concatenate((dataChunk1,dataChunk2,dataChunk3,dataChunk4),axis = 0)
NN1TestingData = dataChunk5
NN2TrainingData = np.concatenate((dataChunk1,dataChunk2,dataChunk3,dataChunk5),axis = 0)
NN2TestingData = dataChunk4
NN3TrainingData = np.concatenate((dataChunk1,dataChunk2,dataChunk4,dataChunk5),axis = 0)
NN3TestingData = dataChunk3
NN4TrainingData = np.concatenate((dataChunk1,dataChunk3,dataChunk4,dataChunk5),axis = 0)
NN4TestingData = dataChunk2
NN5TrainingData = np.concatenate((dataChunk2,dataChunk3,dataChunk4,dataChunk5),axis = 0)
NN5TestingData = dataChunk1


# Train the model

model1 = tf.keras.models.Sequential([Flatten(input_shape=(10, 10)),Dense(128, activation='relu'),Dropout(0.2),Dense(2)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model1.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model2 = tf.keras.models.Sequential([Flatten(input_shape=(10, 10)),Dense(128, activation='relu'),Dropout(0.2),Dense(2)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model2.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model3 = tf.keras.models.Sequential([Flatten(input_shape=(10, 10)),Dense(128, activation='relu'),Dropout(0.2),Dense(2)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model3.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model4 = tf.keras.models.Sequential([Flatten(input_shape=(10, 10)),Dense(128, activation='relu'),Dropout(0.2),Dense(2)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model4.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model5 = tf.keras.models.Sequential([Flatten(input_shape=(10, 10)),Dense(128, activation='relu'),Dropout(0.2),Dense(2)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model5.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])





model1.fit(NN1TrainingData, NN1TrainingLabels, epochs=50, validation_data=(NN1TestingData, NN1TestingLabels), callbacks=[early_stopping])
model2.fit(NN2TrainingData, NN2TrainingLabels, epochs=50, validation_data=(NN2TestingData, NN2TestingLabels), callbacks=[early_stopping])
model3.fit(NN3TrainingData, NN3TrainingLabels, epochs=50, validation_data=(NN3TestingData, NN3TestingLabels), callbacks=[early_stopping])
model4.fit(NN4TrainingData, NN4TrainingLabels, epochs=50, validation_data=(NN4TestingData, NN4TestingLabels), callbacks=[early_stopping])
model5.fit(NN5TrainingData, NN5TrainingLabels, epochs=50, validation_data=(NN5TestingData, NN5TestingLabels), callbacks=[early_stopping])




validation_data, validation_labels, validation_temps = load_and_preprocess_data('ValidationData.dat', 'ValidationLabels.dat', 'ValidationTemps.dat', 2500, 10)

validation_labels = validation_labels.flatten()
validation_temps = validation_temps.flatten()



guesses1 = []

for t in range(5):
    for sample in range(5):
        index = t * 5 + sample
        prediction1 = model1.predict(validation_data[index:index+1])
        guesses1.append(np.argmax(prediction1, axis=1))

guesses1 = np.array(guesses1).flatten()

accuracies = []
temperatures = []

for t in range(5):
    correctGuesses = 0
    for sample in range(5):
        index = t * 5 + sample
        if guesses1[index] == validation_labels[index]:
            correctGuesses += 1
    accuracy = correctGuesses / 50
    accuracies.append(accuracy)
    temperatures.append(validation_temps[t * 50])

    
