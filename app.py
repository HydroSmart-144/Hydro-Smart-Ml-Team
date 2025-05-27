from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Label class hasil pelatihan
label_kelas = {
    0: 'bayam_sakit',
    1: 'bayam_sehat',
    2: 'kangkung_sakit',
    3: 'kangkung_sehat',
    4: 'pakcoy_sakit',
    5: 'pakcoy_sehat',
    6: 'sawi_sakit',
    7: 'sawi_sehat',
    8: 'selada_sakit',
    9: 'selada_sehat'
}

# Load model
model = load_model('model.h5')
model.make_predict_function()

# Fungsi prediksi
def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    label = label_kelas[predicted_class]
    confidence = preds[0][predicted_class] * 100
    return label, confidence

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Silakan subscribe."

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = os.path.join('static', img.filename)
        img.save(img_path)

        label, confidence = predict_label(img_path)

        return render_template("index.html", prediction=label, confidence=confidence, img_path=img_path)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
