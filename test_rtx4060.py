import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time

# CRITICAL FIX: Disable mixed precision
tf.keras.mixed_precision.set_global_policy('float32')

print("="*70)
print("GPU TRAINING TEST - FIXED VERSION")
print("="*70)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}")
if gpus:
    print(f"GPU: {gpus[0]}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Memory growth enabled")

# Create test data
print("\nCreating test data...")
X_train = np.random.random((500, 128, 128, 3)).astype(np.float32)
y_train = np.random.randint(0, 2, 500)
X_val = np.random.random((100, 128, 128, 3)).astype(np.float32)
y_val = np.random.randint(0, 2, 100)

print(f"Train: {X_train.shape}, Labels: {y_train.shape}")

# Build model
print("\nBuilding model...")
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # No dtype specified - uses float32
])

model.compile(
    optimizer=keras.optimizers.Adam(0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"Parameters: {model.count_params():,}")

# Train
print("\n" + "="*70)
print("TRAINING 5 EPOCHS")
print("="*70)
print("If you see 'Epoch 1/5' below, GPU is working!\n")

start = time.time()

try:
    history = model.fit(
        X_train, y_train,
        batch_size=16,  # Smaller batch
        epochs=5,
        validation_data=(X_val, y_val),
        verbose=1,
        workers=1,
        use_multiprocessing=False
    )
    
    elapsed = time.time() - start
    print("\n" + "="*70)
    print(f"✅ SUCCESS! Training took {elapsed:.1f} seconds")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print("="*70)
    print("\n🚀 Your GPU is working perfectly!")
    print("Now run: python train_cnn_quick_fixed.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
