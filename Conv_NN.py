# !unzip robots.zip
# !unzip humans.zip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os



# (Number of Images, px,py,rgb)
x = [] #Images
y = [] # categories


folder_robots = 'robots'
folder_humans = 'humans'
folder_Trees = 'Tree'
name_encode = {'robots':0, 'humans':1,'Trees':2}
def images_to_array(folder,name):
  for image in os.listdir(folder):
    loaded_image = Image.open(os.path.join(folder,image))
    resize_image = Image.Image.resize(loaded_image,[100,100])
    image_array = np.array(resize_image)
    x.append(image_array)
    y.append(name_encode[name])





def show_image(index):
  plt.imshow(x[index])
  plt.show()
  print(y[index])

images_to_array(folder_robots,'robots')
images_to_array(folder_humans,'humans')
images_to_array(folder_humans,'Trees')
x = np.array(x)
y = to_categorical(y, num_classes=3)

show_image(51)

from keras.layers import Activation,Conv2D,Dense,Flatten,MaxPool2D,BatchNormalization,Dropout
from keras.models import Sequential


model = Sequential()
model.add(Conv2D(32,(5,5), padding='same',activation='relu',input_shape=(100,100,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(100,(5,5), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.50))
model.add(Conv2D(100,(5,5), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)


from keras.optimizers import Adam

optimizers =Adam(lr=0.001)
model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['acc'])
h = model.fit(x_train,y_train,batch_size=10,epochs=70,validation_data=(x_test,y_test))

model.save('human_robot_CNN.h5')


plt.plot(h.history['acc'], label = 'Train')
plt.plot(h.history['val_acc'], label = 'Test')
plt.title('human_robot_CNN')
plt.xlabel('Epoch Number')

plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()