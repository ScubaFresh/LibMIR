import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import box

def load_audio(audio_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    
    signal, sample_rate = librosa.load(audio_path, sr=target_sr)
    return signal, sample_rate

def extract_audio_features(signal: np.ndarray, sample_rate: int) -> Dict:
    pbar = tqdm(total=9, desc="Extracting Audio features")

    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
    mel_db_scale = librosa.power_to_db(mel_spectrogram, ref=np.max)
    pbar.update(1)    

    music_tempo, beat_positions = librosa.beat.beat_track(y=signal, sr=sample_rate)
    pbar.update(1)

    pitch_classes = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
    pbar.update(1)
    
    timbre_features = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    pbar.update(1)
    
    brightness = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)[0]
    pbar.update(1)
    raw_volume = librosa.feature.rms(y=signal)[0]    
    pbar.update(1)
    volume_range = np.max(raw_volume) - np.min(raw_volume)
    pbar.update(1)
    volume_envelope = raw_volume if volume_range == 0 else (raw_volume - np.min(raw_volume)) / volume_range
    pbar.update(1)    
    note_onsets = librosa.onset.onset_strength(y=signal, sr=sample_rate)
    pbar.update(1)    
    signal_transitions = librosa.feature.zero_crossing_rate(signal)[0]
    pbar.update(1)

    pbar.close()
    
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

def analyze_music_file(file_path: str, output_dir: str = None, no_plot: bool = False) -> None:
    
    print(f"Starting analysis of: {Path(file_path).name}")
    
    signal, sample_rate = load_audio(file_path)
    features = extract_audio_features(signal, sample_rate)
    
    console = Console()
    table = Table(box=box.DOUBLE, show_header=False, show_edge=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add rows with the analysis results
    table.add_row("Tempo", f"{features['tempo']:.1f} BPM")
    table.add_row("Average Volume", f"{np.mean(features['volume']):.3f}")
    table.add_row("Average Brightness", f"{np.mean(features['brightness']):.1f} Hz")
    table.add_row("Signal Complexity", f"{np.mean(features['transitions']):.3f}")
    
    print("\nMusic Analysis Results:")
    console.print(table)
    
    if not no_plot:
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
    
    analyze_music_file(args.input_file, args.output, args.no_plot)

if __name__ == "__main__":
    main()