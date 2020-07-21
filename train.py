import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

# load data
train_data = pd.read_csv('data/train.csv')
X = train_data.loc[:, train_data.columns != 'label']
Y = train_data['label']

# make 0-1
X = X / 255.0

# reshape to 3 dimension
X = X.values.reshape(-1, 28, 28, 1)

# convert Y to [0,0,1,0,0,0,0,0,0,0] instead of single numbers 0,1,2,3,4 etc.
Y = to_categorical(Y, num_classes=10)

# split training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Create Sequential model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3),padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# create optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile model
model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])

# Train model with data
model.fit(X_train, Y_train, batch_size=86, epochs=100, validation_data=(X_test, Y_test), verbose=2)

# Save model to mymodel.model
model.save("mymodel.model")
print("Saved model to 'mymodel.model'")
