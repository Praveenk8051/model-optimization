import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

# Load MNIST dataset and preprocess
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Only use a small subset of data to run on CPU
SAMPLES = 5000
TEST_SAMPLES = 1000

x_train = x_train[:SAMPLES].astype('float32') / 255.0
y_train = y_train[:SAMPLES]
x_test = x_test[:TEST_SAMPLES].astype('float32') / 255.0
y_test = y_test[:TEST_SAMPLES]

# Reshape for CNN input
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training with {SAMPLES} samples, testing with {TEST_SAMPLES} samples")
print(f"Input shape: {x_train.shape}")

# Create a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
print("Model created. Summary:")
model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
print("\nTraining model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save the original model
model.save('mnist_original.h5')
print("Original model saved as 'mnist_original.h5'")
model.save('mnist_original.keras')
print("Original model saved as 'mnist_original.keras'")