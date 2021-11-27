

import numpy as np

from keras.models import load_model


NN_Predict = load_model('Neural_Net_titanic.h5')

input_parameters = np.array([[1,2,50,1,1,15,1,2]])

result = NN_Predict.predict(input_parameters )
print(result)


if result > 0.50:
	result = 'Survived!!'
else:
	result = 'Not Survived'
	
print(result)