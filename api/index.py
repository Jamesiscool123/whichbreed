from flask import Flask, render_template, request
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
np.set_printoptions(suppress=True)
model = load_model("ai/keras_model.h5", compile=False)
class_names = open("ai/labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def checkimage(image):
    allowed = ['.png', '.jpg', '.jpeg']
    for i in range(len(allowed)):
        if allowed[i] in image.filename:
            return True
    return False

def ai(filename):
    image = Image.open(filename).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score*100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['image']
        if image.filename == '':
            return "No selected image"
        if checkimage(image):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(filename)
            class_name, confidence_score = ai(filename)
            con = sqlite3.connect("databases/breeds.db")
            cur = con.cursor()
            daclass = str(class_name).strip()
            info = cur.execute('SELECT Info FROM Breeds WHERE Breed = ?', (daclass,),).fetchone()
            if info is not None:
                infof = str(info)
                infof = infof.strip('()"')
                infof = infof.rstrip('",')
            else:
                infof = type(daclass)
                print(infof)
                print(daclass)
            return render_template('index.html', class_name=class_name, confidence_score=confidence_score, info=infof, file=filename, daclass=daclass)

if __name__ == "__main__":
    app.run()#(debug=False,host='0.0.0.0')
