from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/how-it-works')
def how_it_works():
    return render_template("how_it_works.html")

@app.route('/documentation')
def documentation():
    return render_template("documentation.html")

@app.route("/predict",methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        prediction = model.predict(scaled_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
            
            # Check for image existence (jpg or png)
            import os
            base_img_path = os.path.join('static', 'images', 'crops')
            
            if os.path.exists(os.path.join(base_img_path, f"{crop.lower()}.png")):
                image_name = f"{crop.lower()}.png"
            elif os.path.exists(os.path.join(base_img_path, f"{crop.lower()}.jpg")):
                 image_name = f"{crop.lower()}.jpg"
            else:
                image_name = "img.jpg" # Fallback if specific crop image missing

            return jsonify({'result': result, 'crop': crop, 'image_name': image_name})
        else:
            return jsonify({'result': "Sorry, we could not determine the best crop to be cultivated with the provided data.", 'crop': None, 'image_name': "img.jpg"})
            
    except Exception as e:
        return jsonify({'error': str(e)})




# python main
if __name__ == "__main__":
    app.run(debug=True)