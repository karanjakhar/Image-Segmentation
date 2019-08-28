from flask import Flask, render_template, request,send_file
import os
from PIL import Image
import io
import numpy as np 
from model import predict_mask

app = Flask(__name__)

app.config['SECRET_KEY'] = 'kj' 



@app.route('/',methods = ['GET','POST'])
def index(): 
	if request.method == 'POST':
		img = request.files['img']
		#return " ".join(str(img.getvalue()))
		#image = Image.open(img)
		#image.show()
		#print(np.array(image).shape)
		print('post here') 
		filename = img.filename
		path = os.path.join('/tmp/imgseg/',filename) 
		img.save(path)
		maskname = predict_mask(path)
		return render_template('index.html',uploaded_img_name = filename,result_img_name = maskname)

	elif request.method == 'GET':
		return render_template('index.html',uploaded_img_name = None)


@app.route('/images/<filename>',methods = ['GET'])
def images(filename):
	print('send here')
	return send_file(os.path.join('/tmp/imgseg/',filename))

if __name__ == '__main__':
	app.debug = True
	app.run()
