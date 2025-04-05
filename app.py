from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# تحميل النموذج من ملف pkl
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            data['pregnancies'],
            data['glucose'],
            data['blood_pressure'],
            data['skin_thickness'],
            data['insulin'],
            data['bmi'],
            data['age'],
            data['diabetes_pedigree_function']
        ]

        # التنبؤ باستخدام النموذج المحمل
        prediction = model.predict([features])

        # تحويل النتيجة إلى int
        result = {'prediction': int(prediction[0])}  # assuming binary outcome (0 or 1)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
