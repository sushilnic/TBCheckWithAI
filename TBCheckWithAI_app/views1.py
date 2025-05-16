import os
from django.shortcuts import render
from .forms import AudioForm
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests

UPLOAD_DIR = 'static/uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# GCP Setup
PROJECT_ID = 'tbcheckwithai'
REGION = 'us-central1'
ENDPOINT = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/hear:predict"

CREDENTIALS = service_account.Credentials.from_service_account_file(
    'tbcheckwithai-c915760d7a17.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
CREDENTIALS.refresh(Request())

def analyze_audio(request):
    result = None
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio = request.FILES['audio_file']
            file_path = os.path.join(UPLOAD_DIR, audio.name)
            with open(file_path, 'wb+') as dest:
                for chunk in audio.chunks():
                    dest.write(chunk)

            # Read and encode audio
            with open(file_path, 'rb') as f:
                audio_bytes = f.read().decode('ISO-8859-1')

            headers = {
                "Authorization": f"Bearer {CREDENTIALS.token}",
                "Content-Type": "application/json"
            }

            payload = {
                "instances": [
                    {"audio_bytes": {"bytes": audio_bytes}}
                ]
            }

            response = requests.post(ENDPOINT, headers=headers, json=payload)
            if response.ok:
                result = response.json()
            else:
                result = {"error": response.text}

    else:
        form = AudioForm()
    return render(request, 'TBCheckWithAI_app/index.html', {'form': form, 'result': result})
