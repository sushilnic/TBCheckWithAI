from django.test import TestCase

# Create your tests here.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

model = tf.saved_model.load(r"C:\\Users\\NIC\\.cache\\huggingface\\hub\\models--google--hear\snapshots\\9b2eb2853c426676255cc6ac5804b7f1fe8e563f")
print(list(model.signatures.keys()))
print(model.signatures['serving_default'].structured_input_signature)
print(model.signatures['serving_default'].structured_outputs)


import tensorflow as tf
import numpy as np
#from keras.layers import TFSMLayer
from keras.src.layers import TFSMLayer
# Load the SavedModel as a TFSMLayer
model_path = r"C:\\Users\\NIC\\.cache\\huggingface\\hub\\models--google--hear\snapshots\\9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
layer = TFSMLayer(model_path, call_endpoint='serving_default')

# Example: Create a dummy audio input — 32000 samples of silence (1 second at 32kHz)
# Replace this with your actual audio input
dummy_input = np.zeros((1, 32000), dtype=np.float32)

# Run inference
output = layer(dummy_input)

# Output is a 512-dimensional embedding
print("Output shape:", output.shape)
print("Output:", output.numpy())