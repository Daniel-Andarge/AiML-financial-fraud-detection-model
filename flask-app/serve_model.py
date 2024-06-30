from flask import Flask, request, jsonify
import pandas as pd
from lime import lime_tabular
import shap
import lime
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
    return ("Welcome to ML based fraud detction")


@app.route('/predict', methods=['POST'])
def detect():
    try:
        json_data = request.get_json()

        df = pd.DataFrame(json_data)

        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
      
        # Feature Engineering
        fg = FeatureEngineering(df)
        featured_df = fg.preprocess()
        if len(featured_df) > 0:
            # Make detection
            result = model.predict(featured_df.values).tolist()
            detection_results = ["Fraud" if res == 1 else "Not Fraud" for res in result]

            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(featured_df)
            shap_summary = shap.summary_plot(shap_values, featured_df, plot_type="bar")

            # LIME Explanation
            lime_explainer = lime_tabular.LimeTabularExplainer(featured_df.values, feature_names=featured_df.columns)
            lime_exp = lime_explainer.explain_instance(featured_df.iloc[0].values, model.predict_proba, num_features=5)
            lime_explanation = lime_exp.as_html()

            return jsonify({'Detection': detection_results, 'SHAP_Explanation': shap_summary, 'LIME_Explanation': lime_explanation})
        else:
            return jsonify({'error': 'No data provided for detection'}), 400

    except KeyError as ke:
        return jsonify({'error': f"Missing key in the input data: {ke}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400
