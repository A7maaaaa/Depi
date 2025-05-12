import os
import logging
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from marshmallow import Schema, fields, ValidationError
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', 'svr_model_top4.pkl')
APP_VERSION = '1.0.0'

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)

# Load your trained model
try:
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# Marshmallow schema for input validation (only top 4 features)
class PredictInputSchema(Schema):
    mrp = fields.Float(required=True, validate=lambda x: x >= 0)
    outlet_type = fields.Str(required=True, validate=lambda x: x in ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
    establishment_year = fields.Int(required=True, validate=lambda x: 1985 <= x <= 2009)
    location_type = fields.Str(required=True, validate=lambda x: x in ['Tier 1', 'Tier 2', 'Tier 3'])

# Serve the HTML file
@app.route('/')
def index():
    try:
        # Use the absolute path to index.html
        file_path = os.path.join(os.getcwd(), 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            logging.info(f"Successfully loaded index.html from {file_path}")
            return render_template_string(html_content)
    except Exception as e:
        logging.error(f"Error loading index.html: {e}")
        return f"<h1>Error loading page: {str(e)}</h1>"

# Serve static files if needed (e.g., CSS, JS)
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

@app.route('/version')
def version():
    return jsonify({'version': APP_VERSION, 'model_path': MODEL_PATH})

def preprocess_input(data):
    df = pd.DataFrame([{
        'Item_MRP': data['mrp'],
        'Outlet_Type': data['outlet_type'],
        'Outlet_Establishment_Year': data['establishment_year'],
        'Outlet_Location_Type': data['location_type']
    }])
    # Encode categorical features
    cat_features = ['Outlet_Type', 'Outlet_Location_Type']
    encoder = OrdinalEncoder()
    df[cat_features] = encoder.fit_transform(df[cat_features])
    # Scale features
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled.values

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        json_data = request.get_json()
        schema = PredictInputSchema()
        validated = schema.load(json_data)
        features = preprocess_input(validated)
        prediction = model.predict(features)
        logging.info(f"Prediction made for input: {validated} -> {prediction[0]}")
        return jsonify({'prediction': float(prediction[0])})
    except ValidationError as ve:
        logging.warning(f"Validation error: {ve.messages}")
        return jsonify({'error': 'Invalid input', 'details': ve.messages}), 400
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request', 'message': str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'message': str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=bool(os.environ.get('DEBUG', True)))
# To run the app, use the command: python app.py