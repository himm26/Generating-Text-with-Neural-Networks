import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.metrics import accuracy
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

raw_text = open("wonderland.txt", 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# creating mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

# print(n_chars)
# print(n_vocab)

# preparing the dataset of input to output pairs
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i: i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([chars_to_int[char] for char in seq_in])
    dataY.append(chars_to_int[seq_out])

n_pattern = len(dataX)

X = np.reshape(dataX, (n_pattern, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

int_to_char = dict((i, c) for i, c in enumerate(chars))

filename = "weights-improvement-40-1.3720.hdf5"

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights(filename)
model.compile(loss="categorical_crossentropy", optimizer='adam')
print(model.summary())
#
# #random seed
# start = np.random.randint(0, len(dataX)-1)
# pattern = dataX[start]
#
# print("Seed:")
# print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
#
# # generate characters
# for i in range(1000):
#     x = np.reshape(pattern, (1, len(pattern), 1))
#     x = x/float(n_vocab)
#     prediction = model.predict(x, verbose=0)
#     index = np.argmax(prediction)
#     result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in pattern]
#     sys.stdout.write(result)
#     # print("\n")
#     pattern.append(index)
#     pattern = pattern[1:len(pattern)]
#
# print("\nDone.")

# preparing test set
seq_length = 100
test_dataX = []
test_dataY = []

for j in range(10000):
    i = np.random.randint(0, len(dataX) - 101)
    test_seq_in = raw_text[i: i + seq_length]
    test_seq_out = raw_text[i + seq_length]
    test_dataX.append([chars_to_int[char] for char in test_seq_in])
    test_dataY.append(chars_to_int[test_seq_out])

n_pattern = len(test_dataX)

# rehshaping X to be [samples, time steps, features]
test_X = np.reshape(test_dataX, (n_pattern, seq_length, 1))

# normalize
test_X = test_X / float(n_vocab)

# one hot encode the output variable
test_y = np_utils.to_categorical(test_dataY)

# loss = model.evaluate(test_X, test_y)
# print(loss)

predict = model.predict(test_X, batch_size=128)

pre_y = []
t_y = []

for y in predict:
    pre_y.append(np.argmax(y))
for y in test_y:
    t_y.append(np.argmax(y))


sum = 0
for i in range(len(pre_y)):
    if pre_y[i] == t_y[i]:
        sum = sum + 1

print(sum*100/len(pre_y))

