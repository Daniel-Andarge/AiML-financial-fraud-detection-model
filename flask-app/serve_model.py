from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from scripts.feature_engineering import FeatureEngineering

app = Flask(__name__)

# Load model
with open('models/fraud_detection_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return ("Welvome to ML based fraud detction")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        json_data = request.get_json()

        df = pd.DataFrame(json_data)

        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
      
        #  FeatureEngineering
        fg = FeatureEngineering(df)
        featured_df = fg.preprocess()
        if len(featured_df) > 0:
            # Make detection
            result = model.predict(featured_df.values).tolist()
            detection_results = ["Fraud" if res == 1 else "Not Fraud" for res in result]
            return jsonify({'Detection': detection_results})
        else:
            return jsonify({'error': 'No data provided for detection'}), 400

    except KeyError as ke:
        return jsonify({'error': f"Missing key in the input data: {ke}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)