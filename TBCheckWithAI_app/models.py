from django.db import models

# Create your models here.

class AudioUpload(models.Model):
    audio_file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)