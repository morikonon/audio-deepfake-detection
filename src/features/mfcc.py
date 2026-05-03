import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    # 1. Read audio (y - signal, sr - sampling rate)
    y, sr = librosa.load(file_path, mono=True, duration=30)

    # 2. Compute standard 6 charactecristics 
    features = {
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rms': np.mean(librosa.feature.rms(y=y)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y))
    }

    # 3. Compute 20 MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        # Fill from mfcc0 to mfcc19
        features[f'mfcc{i+1}'] = np.mean(mfcc[i])

    # 4. Create DataFrame
    expected_columns = [
        'chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
        'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
        'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
        'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
        'mfcc19', 'mfcc20'
    ]

    df_features = pd.DataFrame([features], columns=expected_columns)
    return df_features