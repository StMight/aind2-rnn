import string
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    number_of_char = len(series)

    for i in range(0, number_of_char - window_size, 1):
        sequence_input = series[i: i + window_size]
        sequence_output = series[i + window_size]

        X.append(sequence_input)
        y.append(sequence_output)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    wanted_characters = list(string.ascii_lowercase) + list(punctuation)

    text = [character if character in wanted_characters else ' ' for character in text]
    text = ''.join(text)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    number_of_char = len(text)

    for i in range(0, number_of_char - window_size, step_size):
        if(i + window_size + 1 > number_of_char):
            break

        sequence_input = text[i: i + window_size]
        sequence_output = text[i + window_size]

        inputs.append(sequence_input)
        outputs.append(sequence_output) 

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model
