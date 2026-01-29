"""
Simple script to test the trained model on a single image
Usage: python test_model.py path/to/image.jpg
"""

import sys
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = 'models/solar_panel_efficientnet_optimized.h5'
IMG_SIZE = (224, 224)
CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

def load_and_predict(image_path):
    """
    Load an image and make a prediction.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    try:
        # Load model
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✓ Model loaded successfully!\n")
        
        # Load and preprocess image
        print(f"Loading image from {image_path}...")
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded successfully! Size: {image.size}\n")
        
        # Resize and preprocess
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array.astype(np.float32))
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = CLASSES[predicted_idx]
        
        return predicted_class, confidence, predictions[0]
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find image at {image_path}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None, None, None

def print_results(predicted_class, confidence, all_probs):
    """Pretty print the prediction results."""
    if predicted_class is None:
        return
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\n🎯 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    # Confidence interpretation
    if confidence >= 0.9:
        print("✅ Very High Confidence - Highly reliable prediction")
    elif confidence >= 0.75:
        print("✅ High Confidence - Reliable prediction")
    elif confidence >= 0.5:
        print("⚠️  Moderate Confidence - Consider manual inspection")
    else:
        print("❌ Low Confidence - Manual inspection recommended")
    
    print("\n" + "-"*60)
    print("ALL CLASS PROBABILITIES")
    print("-"*60)
    
    # Sort by probability
    sorted_indices = np.argsort(all_probs)[::-1]
    
    for i, idx in enumerate(sorted_indices):
        class_name = CLASSES[idx]
        prob = all_probs[idx]
        
        # Create bar visualization
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        # Medal for top 3
        medal = ""
        if i == 0:
            medal = "🥇 "
        elif i == 1:
            medal = "🥈 "
        elif i == 2:
            medal = "🥉 "
        
        print(f"{medal}{class_name:20s} {bar} {prob:6.2%}")
    
    print("="*60)

def main():
    """Main function."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
        print("\nExample:")
        print("  python test_model.py data/sample_images/clean_sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    # Make prediction
    predicted_class, confidence, all_probs = load_and_predict(image_path)
    
    # Print results
    print_results(predicted_class, confidence, all_probs)

if __name__ == "__main__":
    main()
