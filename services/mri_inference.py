import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json

# Prepare image for the AI model
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the MRI AI model
def load_mri_model(model_path):
    """Loads the MRI model (PyTorch .pt file)."""
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading MRI model: {e}")
        return None

# Run prediction on the MRI image
def predict_mri(model, image_path, label_map):
    image = Image.open(image_path).convert('RGB')
    input_tensor = PREPROCESS(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    class_name = label_map['class_order'][predicted_idx.item()]
    
    # return class probabilities
    class_probs = {label_map['class_order'][i]: float(probs[0][i]) for i in range(len(label_map['class_order']))}
    
    return {
        "predicted_class": class_name,
        "confidence": float(confidence.item()),
        "probabilities": class_probs,
        "input_tensor": input_tensor 
    }
