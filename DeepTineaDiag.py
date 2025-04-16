import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np

class MultitaskModel(nn.Module):
    """
    Multi-task model that performs both classification and segmentation
    using EfficientNet V2 Medium as the backbone.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_v2_m(weights=None)
        self.features = self.backbone.features
        
        in_channels = self.backbone.features[-1][0].out_channels
        
        # Segmentation head
        self.segment = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=5,
                stride=3,
                padding=2,
                output_padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=1,
                out_channels=1,
                kernel_size=15,
                stride=11,
                padding=7,
                output_padding=6,
                bias=False
            ),
            nn.ReLU()
        )
        
        # Classification head
        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.classifier4c = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_channels, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        """Forward pass through the model"""
        features = self.features(x)
        
        # # Segmentation output
        # segmentation = self.segment(features)
        
        # Classification output
        pooled = self.flatten(features)
        classification = self.classifier4c(pooled.squeeze())
        
        return classification

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a pretrained model from the given path
    
    Args:
        model_path: Path to the model weights
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model in evaluation mode
    """
    model = MultitaskModel()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    model.to(device)
    model.eval()
    
    # Freeze all parameters 
    for param in model.parameters():
        param.requires_grad_(False)
        
    return model

def preprocess_image(image_path, transform=None):
    """
    Load and preprocess an image for model inference
    
    Args:
        image_path: Path to the image file
        transform: Transformation pipeline for the image
        
    Returns:
        Preprocessed image tensor
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(480, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    try:
        raw_img = Image.open(image_path).convert('RGB')
        return transform(raw_img).unsqueeze(0)  # Add batch dimension
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def predict(model, image_tensor, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Run inference with the model
    
    Args:
        model: The model to use for inference
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        
    Returns:
        binary_class: Binary class prediction (0 or 1)
        class_probs_binary: Probabilities for binary classification [p(class 0), p(class 1)]
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():  # Disable gradient computation for inference
        # Handle the multi-output model correctly
        classification = model(image_tensor)
    
    # For single-sample inference (batch size of 1)
    if classification.dim() == 1:
        # Apply softmax for a single sample
        class_probs = torch.nn.functional.softmax(classification, dim=0).cpu().numpy()
        
        # For binary classification from multi-class output
        # Assuming class 0 is the positive class and all others are negative
        positive_prob = class_probs[0]
        negative_prob = 1.0 - positive_prob
        class_probs_binary = [positive_prob, negative_prob]
        
        # Determine the binary class
        binary_class = 1 if negative_prob > positive_prob else 0
    else:
        # For batched inference
        class_probs = torch.nn.functional.softmax(classification, dim=1).cpu().numpy()
        
        # For first sample in batch
        positive_prob = class_probs[0, 0]
        negative_prob = 1.0 - positive_prob
        class_probs_binary = [positive_prob, negative_prob]
        
        # Determine the binary class
        binary_class = 1 if negative_prob > positive_prob else 0
    
    return binary_class, class_probs_binary

if __name__ == "__main__":
    model_path='model_weights.pth'
    """Main function to run the model inference pipeline"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
        
    # Load model
    model = load_model(model_path, device)
            
    # Preprocess image
    image_tensor = preprocess_image('Tinea.jpg')

    # Run inference
    class_pred,class_probs = predict(model, image_tensor, device)
        
    print(f"Class: {class_pred}:{class_probs}")