import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
from services.mri_inference import load_mri_model, predict_mri
from services.xai_mri import get_mri_explanation, get_all_mri_heatmaps

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lifestyle_model.pkl')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'models', 'preprocess_config.json')
MRI_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mri_model.pt')
MRI_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_map.json')

model = None
config = None
mri_model = None
mri_label_map = None
mri_error = None

# Load AI models on startup
def load_resources():
    global model, config, mri_model, mri_label_map, mri_error
    try:
        model = joblib.load(MODEL_PATH)
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        mri_model, mri_error = load_mri_model(MRI_MODEL_PATH)
        with open(MRI_LABEL_MAP_PATH, 'r') as f:
            mri_label_map = json.load(f)
        print("Models and configurations loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "config_loaded": config is not None,
        "mri_model_loaded": mri_model is not None,
        "mri_label_map_loaded": mri_label_map is not None,
        "mri_error": mri_error
    })

# Main function to handle MRI and Lifestyle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or config is None or mri_model is None or mri_label_map is None:
        return jsonify({"error": "One or more models or configurations not loaded"}), 500

    # MRI Validation & Inference
    if 'mri_image' not in request.files:
        return jsonify({"error": "MRI image is required"}), 400
    
    mri_file = request.files['mri_image']
    if mri_file.filename == '':
        return jsonify({"error": "No selected MRI image"}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    ext = mri_file.filename.rsplit('.', 1)[1].lower() if '.' in mri_file.filename else ''
    if ext not in allowed_extensions:
        return jsonify({"error": f"Invalid file type. Allowed: {allowed_extensions}"}), 400

    mri_results = {}
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            mri_file.save(tmp.name)
            tmp_path = tmp.name

        mri_results = predict_mri(mri_model, tmp_path, mri_label_map)
        heatmap_base64 = get_mri_explanation(mri_model, mri_results['input_tensor'], tmp_path)
        all_heatmaps = get_all_mri_heatmaps(mri_model, mri_results['input_tensor'], tmp_path, mri_label_map)
        
        mri_results['heatmap'] = heatmap_base64
        mri_results['all_heatmaps'] = all_heatmaps
        
        os.unlink(tmp_path)
        mri_results.pop('input_tensor', None)
    except Exception as e:
        print(f"MRI Inference Error: {e}")
        return jsonify({"error": f"MRI Inference failed: {str(e)}"}), 500

    # Lifestyle Data Validation
    try:
        lifestyle_data = {}
        errors = []
        
        for col in config['numeric_cols']:
            val = request.form.get(col)
            if val is None:
                errors.append(f"Missing required field: {col}")
            else:
                try:
                    lifestyle_data[col] = float(val)
                except ValueError:
                    errors.append(f"Field {col} must be numeric")

        for col in config['categorical_cols']:
            val = request.form.get(col)
            if val is None:
                errors.append(f"Missing required field: {col}")
            else:
                lifestyle_data[col] = val

        if errors:
            return jsonify({"errors": errors}), 400

        # Model Inference & XAI
        df = pd.DataFrame([lifestyle_data])
        from services.xai_lifestyle import explain_prediction
        
        probs_life = model.predict_proba(df)[0]
        top_features = explain_prediction(model, df, config['class_order'])

        # Combine both models 
        alpha = 0.6
        probs_mri_arr = np.array([mri_results['probabilities'][c] for c in config['class_order']])
        probs_life_arr = np.array(probs_life)
        
        probs_final = (alpha * probs_mri_arr) + ((1 - alpha) * probs_life_arr)
        
        prediction_idx = np.argmax(probs_final)
        final_predicted_class = config['class_order'][prediction_idx]
        final_confidence = float(probs_final[prediction_idx])
        
        final_probs_dict = {config['class_order'][i]: float(probs_final[i]) for i in range(len(probs_final))}

        return jsonify({
            "final_prediction": final_predicted_class,
            "final_confidence": final_confidence,
            "final_probabilities": final_probs_dict,
            "mri_prediction": {
                "predicted_class": mri_results['predicted_class'],
                "confidence": mri_results['confidence'],
                "probabilities": mri_results['probabilities'],
                "heatmap": mri_results['heatmap']
            },
            "lifestyle_prediction": {
                "predicted_class": config['class_order'][np.argmax(probs_life)],
                "confidence": float(np.max(probs_life)),
                "probabilities": {config['class_order'][i]: float(probs_life[i]) for i in range(len(probs_life))}
            },
            "xai": {
                "top_features": top_features,
                "gradcam_base64": mri_results['heatmap'], 
                "all_heatmaps": mri_results['all_heatmaps']
            },
            "alpha": alpha,
            "message": "Multimodal prediction successful"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
