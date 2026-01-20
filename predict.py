
import numpy as np
import cv2
from tensorflow import keras
import sys
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("DEEPFAKE DETECTION - PREDICTION TOOL")
print("="*70)

# Load best model (CNN+CBAM)
print("\nLoading CNN+CBAM model...")
model = keras.models.load_model('deepfake_project/models/cnn_cbam_model.h5')
print("✓ Model loaded")

def predict_image(image_path):
    """Predict if image is real or fake"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load: {image_path}")
            return None
        
        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = model.predict(img, verbose=0)[0][0]
        
        # Interpret
        if prediction > 0.5:
            result = "FAKE"
            confidence = prediction * 100
        else:
            result = "REAL"
            confidence = (1 - prediction) * 100
        
        # Display
        print("\n" + "="*70)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*70)
        
        return result, confidence
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Main
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single image
        image_path = sys.argv[1]
        predict_image(image_path)
    else:
        # Interactive mode
        print("\nEnter image path (or 'quit' to exit):")
        while True:
            path = input("> ").strip().strip('"').strip("'")
            if path.lower() in ['quit', 'exit', 'q']:
                break
            if os.path.exists(path):
                predict_image(path)
            else:
                print(f"❌ File not found: {path}")
            print()
