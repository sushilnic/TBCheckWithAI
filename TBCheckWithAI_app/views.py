import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
#from keras.src.layers import TFSMLayer  # ✅ For Keras 3
from keras.layers import TFSMLayer
# Constants
SAMPLE_RATE = 16000
CLIP_DURATION = 2  # seconds
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ✅ Path to downloaded HEAR model from Hugging Face
#model_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f")

# ✅ Load the SavedModel using TFSMLayer
#model_layer = TFSMLayer(model_path, call_endpoint='serving_default')
# Replace this path with your printed snapshot path
MODEL_PATH = r"C:\\Users\\NIC\\.cache\\huggingface\\hub\\models--google--hear\\snapshots\\9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

#loaded = tf.saved_model.load(MODEL_PATH)
#print(list(loaded.signatures.keys()))
# Load the model using TFSMLayer
saved_model_path = "C:/Users/NIC/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

model_layer = TFSMLayer(saved_model_path, call_endpoint="serving_default")


import tensorflow as tf
from keras.layers import TFSMLayer

saved_model_path = "C:/Users/NIC/.cache/huggingface/hub/models--google--hear/snapshots/9b2eb2853c426676255cc6ac5804b7f1fe8e563f"

model = TFSMLayer(saved_model_path, call_endpoint="serving_default")

def resample_audio_and_convert_to_mono(audio_array, original_rate):
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    new_sample_count = int(round(audio_array.shape[0] * (SAMPLE_RATE / original_rate)))
    return signal.resample(audio_array, new_sample_count)

def simple_tb_rule_classifier(embedding_vector):
    avg_value = np.mean(embedding_vector)
    print(f"Average value of embedding vector: {avg_value}")
    threshold = 0.05
    if avg_value > threshold:
        return f"threshold = 0.05<br>High chance of TB (based on audio pattern: {avg_value})"
    else:
        return f"threshold = 0.05<br>Low chance of TB (based on audio pattern: {avg_value})"

def home(request):
    return render(request, 'TBCheckWithAI_app/index.html')

def analyze_audio(request):
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        file_path = default_storage.save(os.path.join('uploads', audio_file.name), audio_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            sample_rate, audio_array = wavfile.read(full_path)
            audio_array = resample_audio_and_convert_to_mono(audio_array, sample_rate)

            # Clip to 2 seconds
            # padded_audio = np.zeros(CLIP_LENGTH, dtype=np.float32)
            # padded_audio[:min(CLIP_LENGTH, len(audio_array))] = audio_array[:CLIP_LENGTH]
            # #input_tensor = np.expand_dims(padded_audio, axis=0)
            # # After defining model as TFSMLayer
            # input_tensor = np.expand_dims(audio_array[:CLIP_LENGTH], axis=0).astype(np.float32)
            # output = model({"x": input_tensor})  # Input name must match the SavedModel's signature

            # embedding_vector = output['output_0'].flatten()
            # Run inference
            #output = model_layer(tf.constant(input_tensor, dtype=tf.float32))
            #embedding_vector = output.numpy().flatten()
            # Pad or truncate to exactly 32000 samples
            padded_audio = np.zeros(CLIP_LENGTH, dtype=np.float32)
            length = min(len(audio_array), CLIP_LENGTH)
            padded_audio[:length] = audio_array[:length]

            input_tensor = np.expand_dims(padded_audio, axis=0).astype(np.float32)

            output = model(input_tensor)  # ✅ direct tensor input
            #embedding_vector = output['output_0'].flatten()
            embedding_tensor = output['output_0']
            embedding_vector = tf.reshape(embedding_tensor, [-1]).numpy()
            prediction_result = simple_tb_rule_classifier(embedding_vector)


            
            return render(request, 'TBCheckWithAI_app/result.html', {
                'prediction_result': prediction_result
            })

        except Exception as e:
            return render(request, 'TBCheckWithAI_app/result.html', {
                'prediction_result': f"Error during processing: {str(e)}"
            })

    return render(request, 'TBCheckWithAI_app/index.html')
