
import os
import numpy as np
import cv2
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("TESTING ON YOUR DATASET")
print("="*70)

# Load model
print("\nLoading model...")
model = keras.models.load_model('deepfake_project/models/cnn_cbam_model.h5')
print("✓ Model loaded\n")

def predict_image(image_path, true_label):
    """Predict and show result"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        pred = model.predict(img, verbose=0)[0][0]
        predicted = "FAKE" if pred > 0.5 else "REAL"
        confidence = (pred if pred > 0.5 else 1-pred) * 100
        
        correct = (predicted == true_label)
        status = "✅" if correct else "❌"
        
        print(f"{status} True: {true_label:4s} | Pred: {predicted:4s} | Conf: {confidence:5.1f}% | {os.path.basename(image_path)}")
        return correct
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test paths
real_dir = r"C:\Users\mysel\Desktop\deepfake_isl\data\real"
fake_dir = r"C:\Users\mysel\Desktop\deepfake_isl\data\fake"

results = []

# Test REAL images
if os.path.exists(real_dir):
    real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.png'))][:15]
    print(f"Testing {len(real_files)} REAL images:")
    print("-"*70)
    for f in real_files:
        result = predict_image(os.path.join(real_dir, f), "REAL")
        if result is not None:
            results.append(result)
else:
    print(f"❌ Real folder not found: {real_dir}")

print()

# Test FAKE images
if os.path.exists(fake_dir):
    fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.png'))][:15]
    print(f"Testing {len(fake_files)} FAKE images:")
    print("-"*70)
    for f in fake_files:
        result = predict_image(os.path.join(fake_dir, f), "FAKE")
        if result is not None:
            results.append(result)
else:
    print(f"❌ Fake folder not found: {fake_dir}")

# Summary
if results:
    correct = sum(results)
    total = len(results)
    accuracy = correct / total * 100
    
    print("\n" + "="*70)
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("="*70)
    
    # Breakdown
    real_correct = sum(results[:15]) if len(results) >= 15 else sum(results[:len(results)//2])
    fake_correct = sum(results[15:]) if len(results) > 15 else sum(results[len(results)//2:])
    
    print(f"\nReal images: {real_correct}/15 correct")
    print(f"Fake images: {fake_correct}/15 correct")
