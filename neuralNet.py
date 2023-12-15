
#Setting up
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



# Function to look at arrays for troubleshooting
def showMe(x,nameString,override=False):
    print(nameString)

    print(x)
    print('uniques: ', np.unique(x))
    print('length: ', len(x))
    print('size: ', np.size(x))
    print('average: ', np.mean(x))
    

# Load and preprocess data
def load_and_preprocess_data(data_path, labels_path, num_rows, side_length):
    data = np.fromfile(data_path, dtype=np.int32).reshape(num_rows, side_length, side_length)
    labels = np.fromfile(labels_path, dtype=np.int32).reshape(num_rows, 1)
    return data, labels


def shuffle_data(*arrays):
    length = len(arrays[0])
    if not all(len(arr) == length for arr in arrays):
        raise ValueError("All arrays must have the same length")

    indices = np.random.permutation(length)
    return tuple(arr[indices] for arr in arrays)


def getModelData(model):

    model_training_labels = np.concatenate((training_labels[:model*size], training_labels[(model+1)*size:]), axis=0)
    model_testing_labels = training_labels[model*size:(model+1)*size]

    #model_testing_temps = np.concatenate((training_labels[:model*size], training_labels[(model+1)*size:]), axis=0)

    model_training_data = np.concatenate((training_data[:model*size, :, :], training_data[(model+1)*size:, :, :]), axis=0)
    model_testing_data = training_data[model*size:(model+1)*size, :, :]

    return model_training_labels, model_testing_labels, model_training_data, model_testing_data


def get_integers_except_n(start, end, n):
    return [i for i in range(start, end + 1) if i != n]

def create_model():
    model = tf.keras.models.Sequential([
        Flatten(input_shape=(side_length, side_length)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_node_number)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model


# Get parameters of run 

params = np.fromfile('data\\params.dat', dtype=np.float64)
print(params)

dimension = int(params[0])
side_length = int(params[1])
temperature_number = int(params[2])
sample_number = int(params[3])
min_temp = params[4]
max_temp = params[5]

num_rows = temperature_number * sample_number

# Load and preprocess training and validating data
folder_string = 'data\\L=' + str(side_length) + '\\'

training_data = np.fromfile(folder_string+'trainingData.dat', dtype=np.int32).reshape(num_rows, side_length, side_length)
training_labels = np.fromfile(folder_string+'trainingLabels.dat', dtype=np.int32).reshape(num_rows, 1)
training_temps = np.fromfile(folder_string+'trainingTemps.dat', dtype=np.float64).reshape(num_rows, 1)

training_data,training_labels, training_temps = shuffle_data(training_data,training_labels, training_temps)

validating_data = np.fromfile(folder_string+'validatingData.dat', dtype=np.int32).reshape(num_rows, side_length, side_length)
validating_labels = np.fromfile(folder_string+'validatingLabels.dat', dtype=np.int32).reshape(num_rows, 1)
validating_temps = np.fromfile(folder_string+'validatingTemps.dat', dtype=np.float64).reshape(num_rows, 1)
validating_tnumbers = np.fromfile(folder_string+'validatingTNumbers.dat', dtype=np.int32).reshape(num_rows, 1)




# Define the neural network model, here and in function create_model()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)



# We will train 5 models on different parts of the data.
modelNumber = 5
size = num_rows // modelNumber  # Use integer division to ensure an integer size

training_labels = training_labels.flatten()




output_node_number = 2





# Start with an empty array with the right shape, counters for total guesses and correct guesses for each of the models
temperature_total_guesses = np.zeros((modelNumber,temperature_number))
temperature_right_guesses = np.zeros((modelNumber,temperature_number))


# Create models. I'm still holding each model in memory at once, though it is now all done one-model-at-a-time
models = [create_model() for _ in range(modelNumber)]
validating_labels = validating_labels.flatten()

for i, model in enumerate(models):

    # Use getModelData() to define each model's training and validation data
    model_training_labels, model_testing_labels, model_training_data, model_testing_data = getModelData(i)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(model_training_data, model_training_labels, epochs=50, validation_data=(model_testing_data, model_testing_labels), callbacks=[early_stopping])
    

    for t in range(temperature_number):
        for sample in range(sample_number):
            index = t * sample_number + sample

            # tNumber runs from 0 to 49, the number of temperatures chosen, with 0 the coldest and 49 the hottest
            tNumber = validating_tnumbers[index]

            correct_label = validating_labels[index]

            print('Correct label is ',correct_label)
            temperature_total_guesses[i,tNumber] = temperature_total_guesses[i,tNumber] + 1


            
            model_prediction = model.predict(validating_data[index:index+1],verbose=0)
            guessed_label = np.argmax(model_prediction, axis=1)
            print('Model ' + str(i) + ', Temperature ' + str(t) + ', sample ' + str(sample) + ' guessed ' + str(guessed_label))

            if guessed_label == correct_label:
                temperature_right_guesses[i,tNumber] = temperature_right_guesses[i,tNumber] + 1
    

    



# Define and track values of use to generating figure
means = np.zeros(temperature_number)
errors = np.zeros(temperature_number)
accuracy = temperature_right_guesses/temperature_total_guesses




for t in range(temperature_number):
    mean = np.mean(accuracy[:,t])
    error = np.std(accuracy[:,t])/np.sqrt(modelNumber)

    means[t] = mean
    errors[t] = error





temperatures = np.linspace(min_temp,max_temp,temperature_number)
plt.errorbar(temperatures,means,errors)

plt.axhline(y=1, color='black')
plt.axvline(x=2/(np.log(1+np.sqrt(2))), color='black', linestyle='dotted')

plt.title("Accuracy of an Ensemble of 5 Neural Networks \n Classifying Configurations of a 10^2 Ising Lattice")
plt.xlabel("Temperature")
plt.ylim(0, 1.2)  # Set y-axis limits from 0 to 1.2
plt.ylabel("Mean Accuracy of Neural Networks")
plt.show()
