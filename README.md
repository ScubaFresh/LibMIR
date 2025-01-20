# Audio Analysis Tool

A Python script for analyzing audio files using librosa. This tool extracts various musical features and generates visualizations.

## Features

- Mel spectrogram analysis
- Tempo detection
- Beat tracking
- Pitch class analysis
- Timbral feature extraction (MFCCs)
- Volume envelope analysis
- Note onset detection
- Signal transition analysis

## Requirements

```
python >= 3.7
librosa
numpy
matplotlib
```


## Usage

Basic usage:
```
python libmir.py input_audio.mp3
```

With optional arguments:
```
# Save plots to specific directory
python analyze_audio.py input_audio.mp3 -o /path/to/output

# Change sample rate
python analyze_audio.py input_audio.mp3 -sr 44100

# Skip plot generation
python analyze_audio.py input_audio.mp3 --no-plot
```

## Command Line Arguments

- `input_file`: Path to the input audio file (required)
- `-o, --output`: Directory to save plots (optional)
- `-sr, --sample-rate`: Target sample rate (default: 22050 Hz)
- `--no-plot`: Skip plot generation

## Output

The script provides:
1. Numerical analysis results including:
   - Tempo (BPM)
   - Average volume
   - Average spectral brightness
   - Signal complexity

2. Visualization plots (unless --no-plot is used):
   - Mel spectrogram
   - Pitch class distribution
   - Timbral features
   - Volume envelope
   - Signal transitions
   - Note onset strength

## File Format Support

Supports various audio formats including:
- WAV
- MP3
- FLAC
- OGG
- M4A

## License

MIT License
