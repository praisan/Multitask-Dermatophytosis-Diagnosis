# Multitask Deep Learning for Dermatophytosis Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) This repository provides the official PyTorch implementation and pretrained models for the research paper:

**"Deep Learning for Diagnosis of Tinea Corporis and Tinea Cruris: Artificial intelligence for diagnosis of dermatophytosis"**

* **Authors:** Narachai Julanon MD<sup>1</sup>, Anupol Panitchote MD<sup>2</sup>, Prisan Padungweang<sup>3</sup>, Charoen Choonhakarn MD<sup>1</sup>, Suteeraporn Chaowattanapanit MD<sup>1</sup>

<sup>1</sup>Division of Dermatology, Department of Medicine, Faculty of Medicine, Khon Kaen University, Khon Kaen, Thailand

<sup>2</sup>Division of Critical Care Medicine, Department of Medicine, Faculty of Medicine, Khon Kaen University, Khon Kaen, Thailand

<sup>3</sup>College of Computing, Khon Kaen University, Khon Kaen, Thailand

* **Paper:** Deep Learning for Diagnosis of Tinea Corporis and Tinea Cruris


## Overview

This project utilizes a multitask deep learning model for the diagnosis of dermatophytosis, specifically focusing on Tinea Corporis and Tinea Cruris. The model is designed to perform classification based on clinical images.

The core model uses an **EfficientNetV2 Medium** backbone, pretrained on ImageNet, and adapted for the dermatological diagnosis task. It employs a multitask learning approach, although the provided inference code primarily focuses on the classification task.

## Features

* **Multitask Architecture:** Based on EfficientNetV2 Medium, designed for multiple diagnostic outputs (currently implemented for classification).
* **Binary Classification Output:** The provided `predict.py` script converts the 4-class probabilities into a final binary classification:
    * Class 0: Dermatophyte
    * Class 1: Other
* **(Potential) Segmentation:** The model architecture includes a segmentation head (currently commented out in the provided `forward` pass in the inference script). 

* **Download Model Weights:**
    Download the pretrained model weights (`model_weights.pth`) from:  
    [model_weights.pth - Google Drive](https://drive.google.com/file/d/1HmNw-HUnVZiwWpCZKOltnD_gHjHUI0aL/view?usp=drive_link)
    Place the `model_weights.pth` file in the root directory.

## Usage (Inference)

The provided script `predict.py` demonstrates how to load the model and perform inference on a single image.

1.  **Prepare your image:** Make sure you have an image file (e.g., `your_image.jpg`) you want to classify.
2.  **Modify the script:** Update the `image_path` variable in the script to point to your image file. You can adapt the main execution block like this:

    ```python
    # predict.py (main execution block)
    import numpy as np # Make sure numpy is imported

    # --- Include the MultitaskModel, load_model, preprocess_image, and predict functions here ---

       if __name__ == "__main__":
        model_path = 'model_weights.pth' # Make sure this file exists
        image_path = 'path/to/your_image.jpg' # <--- CHANGE THIS TO YOUR IMAGE PATH

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        try:
            # Load model
            print("Loading model...")
            model = load_model(model_path, device)
            print("Model loaded successfully.")

            # Preprocess image
            print(f"Processing image: {image_path}")
            image_tensor = preprocess_image(image_path)
            print("Image preprocessed.")

            # Run inference
            print("Running inference...")
            # The predict function now returns two values
            binary_class, class_probs_binary = predict(model, image_tensor, device)
            print("Inference complete.")

            # Print results
            print("\nBinary Classification Result:")

            # Define the meaning of your binary classes clearly
            binary_class_meaning = {
                0: "Dermatophyte", 
                1: "Other"
            }
            print(f"  Probability (Class 0 - Positive): {class_probs_binary[0]:.4f}")
            print(f"  Probability (Class 1 - Negative): {class_probs_binary[1]:.4f}")
            print(f"  Predicted Binary Class: {binary_class} ({binary_class_meaning[binary_class]})")


        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure the model weights and image file exist at the specified paths.")
        except Exception as e:
            print(f"An error occurred: {e}")

    ```
    ```

3.  **Run the script:**
    ```bash
    python predict.py
    ```

4.  **Output:** The script will print the predicted probabilities for each of the 2 classes.

## Model Details

* **Backbone:** EfficientNetV2 Medium (`models.efficientnet_v2_m`)
* **Input Size:** Images are resized and center-cropped to 480x480 pixels.
* **Normalization:** Standard ImageNet normalization is applied.
* **Classification Head:** An adaptive average pooling layer followed by a dropout layer and a fully connected layer outputs logits for 4 classes and then mapped to 2 classes.
* **Segmentation Head:** Defined using `ConvTranspose2d` layers but currently inactive in the provided `forward` function of the inference script. It is used in the training process.

## Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{TineaDiagnosisMultitask, 
  title   = {Deep Learning for Diagnosis of Tinea Corporis and Tinea Cruris},
  author  = {Narachai Julanon, Anupol Panitchote, Prisan Padungweang, Charoen Choonhakarn, Suteeraporn Chaowattanapanit},
}
