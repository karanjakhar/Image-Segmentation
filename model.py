from keras.models import load_model
from dice_coef_file import dice_coef
import cv2
import matplotlib.pyplot as plt
from keras import backend as k

def predict_mask(image_path):
	model = load_model('unet_best1.model',custom_objects = {'dice_coef':dice_coef})
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image,(256,256))
	img = model.predict(image.reshape(1,256,256,1))
	k.clear_session()
	plt.imshow(img.reshape(256,256),cmap = 'gray')
	#plt.show()
	name = image_path.split('/')[-1].split('.')[0]+'_mask'
	plt.savefig('/tmp/imgseg/{}.jpeg'.format(name))
	return name+'.jpeg'

