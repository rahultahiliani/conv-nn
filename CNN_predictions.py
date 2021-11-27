



from keras.models import load_model
from Conv_NN import np,x,plt, Image

model = load_model('human_robot_CNN.h5')

def prediction(image_path):
	img_loaded = Image.open(image_path)
	img_resize = Image.Image.resize(img_loaded,(100,100))
	
	
	img = (np.array(img_resize)-127.5)/127.5
	img = img.reshape(1,100,100,3)
	prediction = model.predict_classes(img)
	
	if prediction == 0:
		print('This is Robot')
	elif prediction == 1:
		print('This is human my friend')
	else:
		print('I dont know what are you talking about')
	
	plt.imshow(img_resize)
	plt.show()
	

prediction('200.jfif')





