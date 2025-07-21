import tensorflow as tf

# Load your existing Keras model
print("Loading Keras model: facetracker.h5...")
model = tf.keras.models.load_model('facetracker.h5')
print("Model loaded.")

# Create a TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Add optimizations to make it smaller and faster
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Perform the conversion
print("Converting model to TFLite format...")
tflite_model = converter.convert()
print("Conversion complete.")

# Save the new .tflite model file
with open('facetracker.tflite', 'wb') as f:
  f.write(tflite_model)

print("\nSuccessfully saved 'facetracker.tflite'!")