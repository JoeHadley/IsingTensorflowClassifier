import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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

def shuffle_data(data, labels):
    indices = np.random.permutation(len(labels))
    return data[indices], labels[indices]


def getModelData(model):



    model_training_labels = np.concatenate((training_labels[:model*size], training_labels[(model+1)*size:]), axis=0)
    model_testing_labels = training_labels[model*size:(model+1)*size]

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

# Load and preprocess training and validating data
temperature_number = 50
sample_number = 50
side_length = 10
training_data_number = temperature_number * sample_number
min_temp = 1
max_temp = 5
mode = 1 # Mode 1 - binary labels, Mode 2 - temperature labels


print('Got here!')


training_data_string = 'Training'
validating_data_string = 'Validating'

if mode == 1: # Use binary labels 
    training_data, training_labels = load_and_preprocess_data(training_data_string+'Data.dat', training_data_string+'Blabels.dat', training_data_number, side_length)
    training_data, training_labels = shuffle_data(training_data, training_labels)
    validation_data, validation_labels = load_and_preprocess_data(validating_data_string+'Data.dat', validating_data_string+'Blabels.dat', training_data_number, side_length)
    output_node_number = 2

else: # Use temperature labels
    training_data, training_labels = load_and_preprocess_data(training_data_string+'Data.dat', training_data_string+'Tlabels.dat', training_data_number, side_length)
    training_data, training_labels = shuffle_data(training_data, training_labels)
    validation_data, validation_labels = load_and_preprocess_data(validating_data_string+'Data.dat', validating_data_string+'Tlabels.dat', training_data_number, side_length)
    # Recover temperatures from temperature labels
    training_temps = min_temp + training_labels*(max_temp-min_temp)/(temperature_number-1)
    validation_temps = min_temp + validation_labels*(max_temp-min_temp)/(temperature_number-1)
    output_node_number = temperature_number







# Define the neural network model


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)



# We will train 5 models on different parts of the data.
modelNumber = 5
size = training_data_number // modelNumber  # Use integer division to ensure an integer size

training_labels = training_labels.flatten()



# Train the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



#models = [create_model() for _ in range(modelNumber)]

#for i, model in enumerate(models):
#    model_training_labels, model_testing_labels, model_training_data, model_testing_data = getModelData(i)
#    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])





neuralNet = tf.keras.models.Sequential([
    Flatten(input_shape=(side_length, side_length)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(output_node_number)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
neuralNet.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


model = 0

model_training_labels = np.concatenate((training_labels[:model*size], training_labels[(model+1)*size:]), axis=0)
model_testing_labels = training_labels[model*size:(model+1)*size]

model_training_data = np.concatenate((training_data[:model*size, :, :], training_data[(model+1)*size:, :, :]), axis=0)
model_testing_data = training_data[model*size:(model+1)*size, :, :]

print(model_testing_data)


'''
neuralNet.fit(model_training_data, model_training_labels, epochs=50, validation_data=(model_testing_data, model_testing_labels), callbacks=[early_stopping])
'''






validation_labels = validation_labels.flatten()

#showMe(validation_data[600])
'''
print(validation_data[600])
print(validation_labels[600])


print(sum(sum(validation_data[1000]))/100)
'''


'''
critTemp = 2.269185
badRowArray = np.array([])
badLabelArray = np.array([])
badTempArray = np.array([])
threshold = 0.5

dataArray = 2*validation_data - 1
labelArray = validation_labels

if mode == 2:

    tempArray = training_temps



for i in range(training_data_number):
    badness = 0
    badnessString = 'fine'
    magnetization = np.mean(dataArray[i])

    
    if mode == 2:

        if tempArray < critTemp:

            tempString = 'cold'
            
            if abs(magnetization) < threshold:

                badness = 1
                badnessString = 'bad'
                badRowArray = np.append(badRowArray,i)
                badLabelArray = np.append(badLabelArray,labelArray[i])


        elif  tempArray > critTemp:

            tempString = 'hot'

            if abs(magnetization) > threshold:

                badness = 1
                badnessString = 'bad'
                badRowArray = np.append(badRowArray,i)
                badLabelArray = np.append(badLabelArray,labelArray[i])

    else:

        if labelArray[i] == 0:

            tempString = 'cold'
            
            if abs(magnetization) < threshold:

                badness = 1
                badnessString = 'bad'
                badRowArray = np.append(badRowArray,i)
                badLabelArray = np.append(badLabelArray,labelArray[i])


        else:

            tempString = 'hot'

            if abs(magnetization) > threshold:

                badness = 1
                badnessString = 'bad'
                badRowArray = np.append(badRowArray,i)
                badLabelArray = np.append(badLabelArray,labelArray[i])



    showString = 'Row ' + str(i) + ', label ' +  str(labelArray[i]) + ', lattice is '  + tempString + ', mean is ' + str(magnetization) + ', this is ' + badnessString
    print(showString)


#showMe(badLabelArray)
#showMe(badTempArray)


showMe(badRowArray,'badRowArray',True)
print('uniques: ', np.unique(badRowArray))

showMe(dataArray[860],'Validation 860')
print(validation_labels[860])


'''


big_accuracies = np.array([]).reshape(0, temperature_number)  # Start with an empty array with the right shape
temperatures = np.unique(validation_temps)



temperature_total_guesses = np.zeros(temperature_number)
temperature_right_guesses = np.zeros(temperature_number)
temperature_nearly_right_guesses = np.zeros(temperature_number)



#for i, model in enumerate(models):
model_guesses = []
model_accuracies = []

for t in range(temperature_number):
    for sample in range(sample_number):
        index = t * sample_number + sample
        
        correct_label = validation_labels[index]

        print('Correct label is ',correct_label)
        temperature_total_guesses[correct_label] = temperature_total_guesses[correct_label] + 1


        
        model_prediction = model0.predict(validation_data[index:index+1],verbose=0) # sus
        guessed_label = np.argmax(model_prediction, axis=1)
        print('Model ' + str(model) + ', Temperature ' + str(t) + ', sample ' + str(sample) + ' guessed ' + str(guessed_label))

        if guessed_label == correct_label:
            temperature_right_guesses[correct_label] = temperature_right_guesses[correct_label] + 1

print(temperature_right_guesses/temperature_total_guesses)
    








means = []
errors = []

for t in range(temperature_number):
    mean = np.mean(big_accuracies[:,t])
    error = np.std(big_accuracies[:,t])/np.sqrt(modelNumber)

    means.append(mean)
    errors.append(error)
