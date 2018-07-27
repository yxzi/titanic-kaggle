import tensorflow as tf 
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 

import csv

###load data
#with open('train.csv', 'rb') as csvfile:
#	reader = csv.reader(csvfile)
#	for row in reader:
#		print(row)

data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
#print(data)
print(data[0])
data = np.delete(data, 0, 1)
data = np.delete(data, np.s_[2:4], 1)
data = np.delete(data, np.s_[6], 1)
data = np.delete(data, np.s_[7], 1)
print(data[0])

###load data
with open('train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	i = -1
	age = 0
	avg = 0
	for row in reader:
		if i == -1:
			i += 1
			continue

		if not np.isnan(data[i][3]):
			age += data[i][3]
			avg = age / (i + 1)

		if row[4] == 'male':
			sex = 1
		else:
			sex = 0

		if row[11] == "S":
			embark = 1
		elif row[11] == "Q":
			embark = 2
		else:
			embark = 3

		data[i][2] = sex
		data[i][7] = embark
		if np.isnan(data[i][3]):
			data[i][3] = avg
		#print(data[i])

		i += 1

# shuffle the data
#print(data.shape)
order = np.argsort(np.random.random(data.shape[0]))
train_data = data[:,1:]
train_labels = data[:,0]
train_data = train_data[order]
train_labels = train_labels[order]
print(train_data[0])
print(train_labels[0])

test_data = np.genfromtxt('test.csv', delimiter=',', skip_header=1)
#print(data)
print(test_data[0])
test_id = test_data[:,0]
test_data = np.delete(test_data, 0, 1)
test_data = np.delete(test_data, np.s_[1:3], 1)
test_data = np.delete(test_data, np.s_[5], 1)
test_data = np.delete(test_data, np.s_[6], 1)
print(test_data[0])

###load data
with open('test.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	i = -1
	age = 0
	avg = 0
	price = 0
	for row in reader:
		if i == -1:
			i += 1
			continue

		if not np.isnan(test_data[i][2]):
			age += test_data[i][2]
			avg = age / (i + 1)

		if not np.isnan(test_data[i][5]):
			price += test_data[i][5]

		if row[3] == 'male':
			sex = 1
		else:
			sex = 0

		if row[10] == "S":
			embark = 1
		elif row[10] == "Q":
			embark = 2
		else:
			embark = 3

		#print(test_data[i])
		test_data[i][1] = sex
		test_data[i][6] = embark
		if np.isnan(test_data[i][2]):
			test_data[i][2] = avg
		if np.isnan(test_data[i][5]):
			test_data[i][5] = price / i
		#print(avg)
		#print(test_data[i])

		i += 1


# normalize data
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#print((mean, std))
print(train_data[0])
print(test_data[0])


# separate validation data
#x_val = train_data[:300]
#x_train = train_data[300:]

#y_val = train_labels[:300]
#y_train = train_labels[300:]

# build model
def build_model():
	model = keras.Sequential([
		keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
		keras.layers.Dense(16, activation=tf.nn.relu),
		keras.layers.Dense(1, activation=tf.nn.sigmoid)
		])

	optimizer = tf.train.RMSPropOptimizer(0.001)

	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

model = build_model()
#model.summary()

# train model
history = model.fit(train_data, train_labels, epochs=40, validation_split=0.2, verbose=2)

# make predictions
test_predictions = model.predict(test_data)
print((len(test_predictions), len(test_id)))
test_id = np.expand_dims(test_id, axis=1)
#print(test_predictions)
#print(test_id)
#print(test_predictions.shape)
#print(test_id.shape)
test_predictions[test_predictions > 0.5] = 1
test_predictions[test_predictions <= 0.5] = 0
pre = np.concatenate((test_id, test_predictions), axis=1)

np.savetxt('result.csv', pre, fmt='%d', delimiter=',', header='PassengerId,Survived', comments='')
# write to csv file
#with open('result.csv', 'w') as csvfile:
#	result_writer = csv.writer(csvfile, delimiter=',')
##	for i in range(len(test_id)):
#		if test_predictions[i] > 0.5:
#			p = 1
#		else:
#			p = 0
#		result_writer.writerow([int(test_id[i]), p])

#print(test_predictions)
