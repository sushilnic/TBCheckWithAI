import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from scipy.io import wavfile
from scipy import signal
from huggingface_hub import from_pretrained_keras
import tensorflow as tf
from huggingface_hub import login

login(token="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")
# Constants
SAMPLE_RATE = 16000
CLIP_DURATION = 2  # seconds
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION

# Load HEAR model only once
loaded_model = from_pretrained_keras("google/hear")

def resample_audio_and_convert_to_mono(audio_array, original_rate):
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    new_sample_count = int(round(audio_array.shape[0] * (SAMPLE_RATE / original_rate)))
    return signal.resample(audio_array, new_sample_count)

def simple_tb_rule_classifier(embedding_vector):
    avg_value = np.mean(embedding_vector)
    threshold = 0.05  # Adjust based on your test samples
    if avg_value > threshold:
        return "High chance of TB (based on audio pattern)"
    else:
        return "Low chance of TB (based on audio pattern)"

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

            # Extract segment (start at 0)
            input_tensor = np.expand_dims(audio_array[:CLIP_LENGTH], axis=0)
            output = loaded_model.signatures["serving_default"](x=tf.constant(input_tensor, dtype=tf.float32))
            embedding_vector = output['output_0'].numpy().flatten()

            # Rule-based prediction
            prediction_result = simple_tb_rule_classifier(embedding_vector)

            return render(request, 'TBCheckWithAI_app/result.html', {
                'prediction_result': prediction_result
            })

        except Exception as e:
            return render(request, 'TBCheckWithAI_app/result.html', {
                'prediction_result': f"Error during processing: {str(e)}"
            })

    return render(request, 'TBCheckWithAI_app/index.html')
