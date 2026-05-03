import numpy as np
import librosa

def audio_to_mel_spectogram(
	path: str,
	sample_rate: int = 16000,
	duration: float = 4.0,
	n_mels: int = 4
):
	y, sr = librosa.load(path, sr=sample_rate, mono=True)

	target_len = int(sample_rate * duration)

	if len(y) < target_len:
		y = np.pad(y, (0, target_len - len(y)))
	else:
		y = y[:target_len]
	
	mel = librosa.feature.melspectogram(
		y=y,
		sr=sr,
		n_mels=n_mels,
		n_fft=1024,
		hop_lenght=256,
	)

	mel_db = librosa.power_to_db(mel, ref=np.max)
	mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8) 
	return mel_db.astype(np.float32)