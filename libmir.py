import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os
from pathlib import Path
import argparse

def load_audio(audio_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    
    signal, sample_rate = librosa.load(audio_path, sr=target_sr)
    return signal, sample_rate

def extract_audio_features(signal: np.ndarray, sample_rate: int) -> Dict:
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    mel_db_scale = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    music_tempo, beat_positions = librosa.beat.beat_track(y=signal, sr=sample_rate)
    
    pitch_classes = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    
    timbre_features = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    
    brightness = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    raw_volume = librosa.feature.rms(y=signal)[0]    
    volume_range = np.max(raw_volume) - np.min(raw_volume)
    volume_envelope = raw_volume if volume_range == 0 else (raw_volume - np.min(raw_volume)) / volume_range
    
    note_onsets = librosa.onset.onset_strength(y=signal, sr=sample_rate)
    
    signal_transitions = librosa.feature.zero_crossing_rate(signal)[0]
    
    return {
        'mel_spectrogram': mel_db_scale,
        'tempo': music_tempo,
        'beats': beat_positions,
        'pitch_content': pitch_classes,
        'timbre': timbre_features,
        'brightness': brightness,
        'volume': volume_envelope,
        'onsets': note_onsets,
        'transitions': signal_transitions
    }

def create_analysis_plots(features: Dict, sample_rate: int, save_path: str = None) -> None:
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    librosa.display.specshow(features['mel_spectrogram'], sr=sample_rate, 
                           x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Frequency Content Over Time')
    
    plt.subplot(3, 2, 2)
    librosa.display.specshow(features['pitch_content'], y_axis='chroma', 
                           x_axis='time')
    plt.colorbar()
    plt.title('Pitch Class Distribution')
    
    plt.subplot(3, 2, 3)
    librosa.display.specshow(features['timbre'], x_axis='time')
    plt.colorbar()
    plt.title('Timbral Features')
    
    plt.subplot(3, 2, 4)
    plt.plot(features['volume'])
    plt.title('Volume Envelope')
    plt.xlabel('Time (frames)')
    
    plt.subplot(3, 2, 5)
    plt.plot(features['transitions'])
    plt.title('Signal Transitions')
    plt.xlabel('Time (frames)')
    
    plt.subplot(3, 2, 6)
    plt.plot(features['onsets'])
    plt.title('Note Onset Strength')
    plt.xlabel('Time (frames)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_music_file(file_path: str, output_dir: str = None) -> None:
    print(f"Starting analysis of: {Path(file_path).name}")
    
    signal, sample_rate = load_audio(file_path)
    features = extract_audio_features(signal, sample_rate)
    
    print("\nMusic Analysis Results:")
    print(f"╔{'═' * 40}╗")
    print(f"║ Tempo: {features['tempo']:.1f} BPM {' ' * 24}║")
    print(f"║ Average Volume: {np.mean(features['volume']):.3f} {' ' * 19}║")
    print(f"║ Average Brightness: {np.mean(features['brightness']):.1f} Hz {' ' * 12}║")
    print(f"║ Signal Complexity: {np.mean(features['transitions']):.3f} {' ' * 15}║")
    print(f"╚{'═' * 40}╝")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{Path(file_path).stem}_analysis.png")
        create_analysis_plots(features, sample_rate, plot_path)
        print(f"\nAnalysis plots saved to: {plot_path}")
    else:
        create_analysis_plots(features, sample_rate)

def main():
    parser = argparse.ArgumentParser(description='Analyze audio files using librosa')
    parser.add_argument('input_file', type=str, help='Path to the input audio file')
    parser.add_argument('-o', '--output', type=str, 
                       help='Output directory for saving plots (optional)')
    parser.add_argument('-sr', '--sample-rate', type=int, default=22050,
                       help='Target sample rate (default: 22050 Hz)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    args = parser.parse_args()
    
    if not args.no_plot:
        analyze_music_file(args.input_file, args.output)
    else:
        signal, sr = load_audio(args.input_file, args.sample_rate)
        features = extract_audio_features(signal, sr)
        print("\nMusic Analysis Results:")
        print(f"╔{'═' * 40}╗")
        print(f"║ Tempo: {features['tempo']:.1f} BPM {' ' * 24}║")
        print(f"║ Average Volume: {np.mean(features['volume']):.3f} {' ' * 19}║")
        print(f"║ Average Brightness: {np.mean(features['brightness']):.1f} Hz {' ' * 12}║")
        print(f"║ Signal Complexity: {np.mean(features['transitions']):.3f} {' ' * 15}║")
        print(f"╚{'═' * 40}╝")

if __name__ == "__main__":
    main()
