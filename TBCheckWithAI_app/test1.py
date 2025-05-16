from huggingface_hub import snapshot_download
model_path = snapshot_download("google/hear")
print(model_path)

import tensorflow as tf

MODEL_PATH = r"C:\\Users\\NIC\\.cache\\huggingface\\hub\\models--google--hear\\snapshots\\9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

loaded = tf.saved_model.load(MODEL_PATH)
print(list(loaded.signatures.keys()))