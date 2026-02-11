import torch
import os
import json
from services.mri_inference import load_mri_model, predict_mri
from services.xai_mri import get_mri_explanation
from PIL import Image

def verify():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'mri_model.pt')
    label_map_path = os.path.join(base_path, 'models', 'label_map.json')
    
    # Check if models exist
    if not os.path.exists(model_path):
        print(f"FAILED: Model not found at {model_path}")
        return
    
    # Load model
    print("Loading model...")
    model = load_mri_model(model_path)
    if model is None:
        print("FAILED: Model loading returned None")
        return
    print("Model loaded successfully.")
    
    # Load label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    print("Label map loaded.")

    # Find a sample image
    sample_dir = os.path.join(os.path.dirname(base_path), 'Sample images')
    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not sample_images:
        print("No sample images found to test.")
        return
    
    test_image = os.path.join(sample_dir, sample_images[0])
    print(f"Testing with image: {test_image}")

    # Run Prediction
    print("Running prediction...")
    results = predict_mri(model, test_image, label_map)
    print(f"Prediction result: {results['predicted_class']} (Confidence: {results['confidence']:.2f})")

    # Run XAI
    print("Running XAI (Grad-CAM)...")
    heatmap_b64 = get_mri_explanation(model, results['input_tensor'], test_image)
    if heatmap_b64:
        print("XAI heatmap generated successfully (Base64 length: {})".format(len(heatmap_b64)))
    else:
        print("FAILED: XAI heatmap generation failed.")

if __name__ == "__main__":
    verify()
