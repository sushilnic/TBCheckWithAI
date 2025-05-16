from django.urls import path
from . import views

# urlpatterns = [
#     path('', views1.analyze_audio, name='analyze_audio'),
# ]


urlpatterns = [
    path('', views.analyze_audio, name='analyze'),
    path('analyze/', views.analyze_audio, name='analyze'),
]