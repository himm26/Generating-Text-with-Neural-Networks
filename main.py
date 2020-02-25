import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# from nltk.corpus import stopwords

raw_text = open("wonderland.txt", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# creating mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))

# print(chars, chars_to_int)


n_chars = len(raw_text)
n_vocab = len(chars)

# print(n_chars)
# print(n_vocab)

# preparing the dataset of input to output pairs
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length,1):
    seq_in = raw_text[i: i+ seq_length]
    seq_out = raw_text[i+ seq_length]
    dataX.append([chars_to_int[char] for char in seq_in])
    dataY.append(chars_to_int[seq_out])
    
n_pattern = len(dataX)


# rehshaping X to be [samples, time steps, features]
X = np.reshape(dataX, (n_pattern, seq_length, 1))

# normalize
X = X/float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)


#LSTM model

model = Sequential()
model.add(LSTM(256, input_shape= (X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation= 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# define the checkpoint

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)



