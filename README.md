
# Speech Emotion Recognition Model

## Model Information
- **Test Accuracy**: 73.61%
- **Number of Emotions**: 8
- **Emotions**: angry, calm, disgust, fearful, happy, neutral, sad, surprised
- **Architecture**: CNN (Convolutional Neural Network)
- **Input**: MFCC features from audio

## Files Included
1. `emotion_recognition_model.keras` - Trained model
2. `label_encoder.pkl` - Label encoder for emotion classes
3. `normalization_params.json` - Mean and std for feature normalization
4. `model_config.json` - Model configuration details
5. `deployment_code.py` - Ready-to-use prediction code

## How to Use

### Installation
```bash
pip install tensorflow librosa numpy
```

### Quick Start
```python
from deployment_code import predict_emotion

emotion, confidence, probabilities = predict_emotion("your_audio.wav")
print(f"Predicted Emotion: {emotion}")
print(f"Confidence: {confidence:.2%}")
```

## Model Performance
- Training completed in 47 epochs
- Best validation accuracy: 77.08%
- Test accuracy: 73.61%

## Requirements
- Python 3.7+
- TensorFlow 2.x
- librosa
- numpy
- scikit-learn

## Notes
- Audio files should be in WAV, MP3, or similar formats
- Model expects audio of approximately 3 seconds
- Works best with clear speech recordings

Created using Google Colab
Date: 2025-11-18

Acknowledgment
This project was completed under the supervision of 👨‍🏫 Anwar Ali Sathio.
