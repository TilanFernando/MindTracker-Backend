import requests
import os
import json

def test_fusion():
    url = "http://127.0.0.1:5000/predict"
    
    # Prepare MRI image
    base_path = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(os.path.dirname(base_path), 'Sample images')
    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not sample_images:
        print("No sample images found.")
        return
    
    image_path = os.path.join(sample_dir, sample_images[0])
    
    # Prepare Lifestyle data (based on numeric_cols and categorical_cols in preprocess_config.json)
    lifestyle_data = {
        "age": "89",
        "education_years": "12",
        "physical_activity_level": "3",
        "sleep_hours": "6",
        "diet_quality": "3",
        "alcohol_intake": "0",
        "social_engagement": "4",
        "cognitive_activity": "3",
        "bmi": "22.3",
        "hypertension": "1",
        "diabetes": "1",
        "cardiovascular_disease": "1",
        "depression_score": "12",
        "family_history_AD": "1",
        "gender": "Male",
        "smoking_status": "Never"
    }
    
    # Send multipart request
    print(f"Sending request to {url} with image {image_path}...")
    with open(image_path, 'rb') as img_file:
        files = {'mri_image': img_file}
        response = requests.post(url, data=lifestyle_data, files=files)
    
    # Parse response
    if response.status_code == 200:
        data = response.json()
        print("\nSUCCESS: Prediction received.")
        print(f"Final Prediction: {data['final_prediction']} (Confidence: {data['final_confidence']:.2f})")
        print(f"MRI Prediction: {data['mri_prediction']['predicted_class']} (Confidence: {data['mri_prediction']['confidence']:.2f})")
        print(f"Lifestyle Prediction: {data['lifestyle_prediction']['predicted_class']} (Confidence: {data['lifestyle_prediction']['confidence']:.2f})")
        print(f"XAI Grad-CAM length: {len(data['xai']['gradcam_base64'])}")
        print(f"XAI Top Features: {[f['feature'] for f in data['xai']['top_features']]}")
    else:
        print(f"\nFAILED: Status {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_fusion()
