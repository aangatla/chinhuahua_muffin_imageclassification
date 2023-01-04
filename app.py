import numpy as np
import os

# Keras
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect,flash, url_for, request, render_template




app = Flask(__name__)



model_path = "model_vgg19.h5"

model = load_model(model_path)

def model_predict(img_path, model):
    img = image.load_img(img_path,target_size = (224,224))
    x = image.img_to_array(img)
    x = x/255.0
    x = x.reshape(1,224,224,3)

    preds = model.predict(x)
    preds = np.argmax(preds,axis = 1)
    if preds == 0:
        preds = "This is a chinhuahua"
    if preds == 1:
        preds = "This is a muffin"
    return preds

@app.route("/", methods = ['GET'])
def index():
    return render_template('index.html')



@app.route("/upload",methods = ["GET","POST"])
def upload():
    if request.method == 'POST':
        img = request.files["file"]
        img_path = 'static/img' + img.filename
        img.save(img_path)
        preds = model_predict(img_path, model)
        result = preds

        return render_template("predict.html", result = result)
    result = "Please upload the image again"
    return render_template("predict.html", result = result )





if __name__ == "__main__":
    app.run()

