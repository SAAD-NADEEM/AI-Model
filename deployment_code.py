
# EMOTION RECOGNITION - DEPLOYMENT CODE
# ======================================
# Use this code to load and run predictions with your trained model

import numpy as np
import librosa
import tensorflow as tf
import pickle
import json

# Load model and artifacts
model = tf.keras.models.load_model('emotion_recognition_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('normalization_params.json', 'r') as f:
    norm_params = json.load(f)
    mean = np.array(norm_params['mean'])
    std = np.array(norm_params['std'])

def extract_mfcc(file_path, max_pad_len=174):
    """Extract MFCC features from audio"""
    audio, sr = librosa.load(file_path, duration=3, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    # Extract features
    mfcc = extract_mfcc(audio_path)
    mfcc_reshaped = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    mfcc_normalized = (mfcc_reshaped - mean) / std
    
    # Predict
    predictions = model.predict(mfcc_normalized, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    emotion = label_encoder.classes_[predicted_class]
    confidence = predictions[predicted_class]
    
    return emotion, confidence, predictions

# Example usage:
# emotion, confidence, probs = predict_emotion("your_audio.wav")
# print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
