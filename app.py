import sys
import io
from flask import Flask, request, jsonify, render_template
import pandas as pd
from tensorflow.keras.models import load_model
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# Load the model
model = load_model('mental_health_model.h5')

# Min and max values for scaling
min_max_values = {
    'GEN_005': (0, 5),
    'GEN_015': (0, 5),
    'GEN_020': (0, 5),
    'GEN_025': (0, 5),
    'GENDVHDI': (0, 5),
    'GEN_010': (0, 10),
    'GEN_030': (0, 5),
    'ALC_015': (0, 10),
    'ALC_020': (0, 10),
    'ALW_005': (0, 10),
    'ALWDVWKY': (0, 1000),
    'ALWDVDLY': (0, 1000),
    'ALWDVLTR': (0, 10),
    'ALWDVSTR': (0, 10),
    'DRGDVYA': (0, 10),
    'DHH_SEX': (1, 2),
    'DHHGAGE': (1, 5),
    'DHHGMS': (1, 5),
    'DHHDGHSZ': (1, 10),
    'DOALW': (0, 10)
}

def normalize_input(input_data, min_max_values):
    normalized_data = {}
    for feature, value in input_data.items():
        min_val, max_val = min_max_values[feature]
        normalized_data[feature] = (value - min_val) / (max_val - min_val)
    return normalized_data

def map_prediction_to_category(prediction):
    categories = {
        0: 'Poor',
        1: 'Poor',
        2: 'Fair',
        3: 'Good',
        4: 'Excellent',
        9: 'Not stated'
    }
    max_index = prediction.argmax()
    return categories.get(max_index, 'Unknown')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        input_data = {k: float(v) for k, v in input_data.items()}
        normalized_data = normalize_input(input_data, min_max_values)

        # Verify that the input data contains all the expected features
        expected_features = list(min_max_values.keys())
        input_df = pd.DataFrame([normalized_data], columns=expected_features)

        # Check the shape of the input DataFrame
        if input_df.shape[1] != len(expected_features):
            raise ValueError(f"Input data must have {len(expected_features)} features, but got {input_df.shape[1]} features.")

        prediction = model.predict(input_df)
        gendvmhi_category = map_prediction_to_category(prediction[0])
        return jsonify({'prediction': gendvmhi_category})
    except Exception as e:
        error_message = traceback.format_exc()
        print(error_message)
        return jsonify({'error': str(e), 'trace': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
