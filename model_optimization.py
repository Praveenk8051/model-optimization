import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_model

# Load the saved model
print("Loading original model...")
model = load_model('mnist_original.h5')

# Load test data for evaluation
print("Loading test data...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
TEST_SAMPLES = 1000
x_test = x_test[:TEST_SAMPLES].astype('float32') / 255.0
y_test = y_test[:TEST_SAMPLES]
x_test = np.expand_dims(x_test, -1)

def evaluate_model(model, description):
    """Evaluate model performance and size"""
    # Performance evaluation
    start_time = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    inference_time = time.time() - start_time
    
    # Size evaluation
    model.save(f'mnist_{description}.h5')
    model_size = os.path.getsize(f'mnist_{description}.h5') / (1024 * 1024)  # MB
    
    print(f"\n{description.upper()} MODEL:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Inference time for {TEST_SAMPLES} samples: {inference_time:.2f} seconds")
    
    return test_acc, model_size, inference_time

# Evaluate original model
original_acc, original_size, original_time = evaluate_model(model, "original")

# ===== OPTIMIZATION METHOD 1: PRUNING =====
print("\n===== APPLYING PRUNING =====")
# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning to the model
pruning_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
pruning_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the pruned model with a few epochs to adapt to pruning
print("Fine-tuning pruned model...")
pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruning_model.fit(
    x_test, y_test,  # Use test data for quick demo
    batch_size=128,
    epochs=2,
    callbacks=pruning_callbacks,
    verbose=1
)

# Strip pruning for final model
final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruning_model)
pruned_acc, pruned_size, pruned_time = evaluate_model(final_pruned_model, "pruned")

# ===== OPTIMIZATION METHOD 2: QUANTIZATION =====
print("\n===== APPLYING QUANTIZATION =====")
# Define the quantization configuration
quantization_aware_model = quantize_model(model)
quantization_aware_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the quantized model with a few epochs to adapt to quantization
print("Fine-tuning quantized model...")
quantization_aware_model.fit(
    x_test, y_test,  # Use test data for quick demo
    batch_size=128,
    epochs=2,
    verbose=1
)

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(quantization_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open('mnist_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

# Check size of TFLite model
tflite_size = os.path.getsize('mnist_quantized.tflite') / (1024 * 1024)  # MB

print("\nQUANTIZED MODEL (TFLite):")
print(f"Model size: {tflite_size:.2f} MB")
print(f"Size reduction: {(1 - tflite_size/original_size) * 100:.2f}%")

# ===== SUMMARY =====
print("\n===== OPTIMIZATION RESULTS SUMMARY =====")
print(f"Original model: {original_size:.2f} MB, {original_acc:.4f} accuracy, {original_time:.2f}s inference")
print(f"Pruned model:   {pruned_size:.2f} MB, {pruned_acc:.4f} accuracy, {pruned_time:.2f}s inference")
print(f"Quantized model (TFLite): {tflite_size:.2f} MB, Size reduction: {(1 - tflite_size/original_size) * 100:.2f}%")
print("\nNOTE: For deployment on CPU, TFLite models can offer significant efficiency gains")