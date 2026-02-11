import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO

# Generate heatmap highlights for the MRI
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap)
        
        return heatmap

def apply_heatmap(heatmap, original_image_path):
    """Superimposes the heatmap on the original image and returns as base64."""
    img = cv2.imread(original_image_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)
    
    _, buffer = cv2.imencode('.png', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

def get_mri_explanation(model, input_tensor, original_image_path, target_layer_name=None):
    """Attempts to find a target layer and generate a Grad-CAM explanation."""
    target_layer = None
    if target_layer_name:
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
    else:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        return None

    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor)
    return apply_heatmap(heatmap, original_image_path)

def get_all_mri_heatmaps(model, input_tensor, original_image_path, label_map, target_layer_name=None):
    """Generates heatmaps for all classes in the label map."""
    target_layer = None
    if target_layer_name:
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break
    else:
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        return {}

    grad_cam = GradCAM(model, target_layer)
    heatmaps = {}
    for i, class_name in enumerate(label_map['class_order']):
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=i)
        heatmaps[class_name] = apply_heatmap(heatmap, original_image_path)
        
    return heatmaps
