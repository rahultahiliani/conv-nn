from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_test.shape)
print(x_train.shape)
plt.imshow(x_train[5], cmap='gray')
plt.show()

reshape = 784
x_train = x_train.reshape(60000,reshape)
x_test = x_test.reshape(10000,reshape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 225
x_test /= 225
print(x_train.shape[1])

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

Ynew_train = np_utils.to_categorical(y_train,10)
Ynew_test = np_utils.to_categorical(y_test,10)

print(Ynew_train)


my_model = Sequential()
my_model.add(Dense(10, input_shape = (784,)))
my_model.add(Dense(10, activation='relu'))
my_model.add(Activation('softmax'))
my_model.summary()


my_model.compile(loss = 'categorical_crossentropy',optimizer=SGD(),metrics=['acc'])
model_train = my_model.fit(x_train,Ynew_train, batch_size=100,epochs=50,verbose=1,validation_split=0.05)
plt.plot(model_train.history['acc'], label = 'Train')
plt.plot(model_train.history['val_acc'], label = 'Test')
plt.xlabel('Epoc Number')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.show()












