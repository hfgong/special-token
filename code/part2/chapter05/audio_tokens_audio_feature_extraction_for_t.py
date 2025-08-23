"""
Audio feature extraction for token generation

Extracted from: part2/chapter05/audio_tokens.tex
Block: 1
Lines: 78
"""

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Mel-spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # MFCC transform for speech
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        
        # Chroma features for music
        self.chroma = T.ChromaScale(
            sample_rate=sample_rate,
            n_chroma=12
        )
        
    def forward(self, waveform, feature_type='mel'):
        """Extract audio features based on specified type."""
        
        if feature_type == 'mel':
            # Mel-spectrogram (general audio)
            mel_spec = self.mel_spectrogram(waveform)
            features = torch.log(mel_spec + 1e-8)  # Log-mel features
            
        elif feature_type == 'mfcc':
            # MFCC (speech processing)
            features = self.mfcc(waveform)
            
        elif feature_type == 'chroma':
            # Chroma (music analysis)
            features = self.chroma(waveform)
            
        elif feature_type == 'combined':
            # Multi-feature representation
            mel_spec = torch.log(self.mel_spectrogram(waveform) + 1e-8)
            mfcc_features = self.mfcc(waveform)
            chroma_features = self.chroma(waveform)
            
            # Concatenate features along frequency dimension
            features = torch.cat([mel_spec, mfcc_features, chroma_features], dim=1)
        
        # Transpose to (batch, time, frequency) for transformer processing
        features = features.transpose(-2, -1)
        
        return features

def preprocess_audio_batch(audio_files, target_length=1000):
    """Preprocess batch of audio files for token generation."""
    
    feature_extractor = AudioFeatureExtractor()
    processed_features = []
    
    for audio_file in audio_files:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Extract features
        features = feature_extractor(waveform, feature_type='combined')
        
        # Pad or truncate to target length
        current_length = features.shape[1]
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            features = F.pad(features, (0, 0, 0, padding))
        elif current_length > target_length:
            # Truncate
            features = features[:, :target_length, :]
        
        processed_features.append(features)
    
    return torch.stack(processed_features)