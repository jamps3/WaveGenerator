# Wave Generator

A Python-based wave generator with modular audio backends, featuring professional-quality audio output, custom waveform drawing, and real-time visualization. Includes both GUI and console interfaces for maximum flexibility.

## Features

Development checkpoint (2026-10-19): the Kivy migration is unfinished and is
preserved on a WIP branch, not promoted as a stable replacement for master.
Nine over-indented control labels were corrected. All root Python sources pass
`python -B -m unittest test_source_syntax -v`, and the GUI source compiles.
These checks do not import GUI/audio/keyboard backends or test sound output.
Legacy Pygame handlers and duplicate methods still need reconciliation, and the
old local Python 3.11 virtual environment references a missing interpreter.
Rebuild a suitable runtime and validate GUI/audio before treating it as usable.

- **Multiple Wave Types**: Sine, square, sawtooth, and custom waveforms
- **Frequency Control**: Adjustable frequency with customizable step sizes
- **Volume Control**: 0-100% volume adjustment
- **Frequency Sweep**: Sweep through frequency ranges with customizable duration and looping
- **Custom Waveform Drawing**: Draw your own waveforms using mouse input (GUI version)
- **Real-time Visualization**: Live waveform display with animation
- **Dual Interfaces**: Choose between console-based or GUI-based operation

## Architecture

The wave generator uses a modular architecture for maximum flexibility:

- **`wave_engine.py`** - Core wave generation engine with smooth transitions and anti-click protection
- **`professional_tone_generator.py`** - Professional audio backend using sounddevice for low-latency, high-quality audio
- **`simple_tone_generator.py`** - Fallback audio backend using pygame for basic functionality
- **`wave_generator.py`** - Main GUI application combining the engine with graphical interface
- **`wave_generator_console.py`** - Console application using the same engine with text-based interface

## Files

- `wave_generator.py` - Main GUI interface using Kivy with professional audio
- `wave_generator_console.py` - Console-based interface for text-based operation
- `wave_engine.py` - Shared wave generation engine with smooth transitions
- `professional_tone_generator.py` - High-quality audio generator using sounddevice
- `simple_tone_generator.py` - Basic audio generator using pygame
- `requirements.txt` - Python dependencies
- `key_test.py` - Keyboard input testing utility

## Installation

1. Install Python 3.11.9 (required for Kivy GUI)
2. Create a virtual environment:
   ```bash
   python3.11 -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GUI Version (Migration In Progress)
```bash
python wave_generator.py
```
Features professional audio output, real-time visualization, and custom waveform drawing.

### Console Version
```bash
python wave_generator_console.py
```
Text-based interface for basic operation without GUI dependencies.

## Controls

### Console Version
- **Arrow Keys**: Navigate menu options
- **Left/Right**: Adjust selected parameter
- **Space**: Toggle play/pause
- **S**: Toggle frequency sweep
- **Q**: Quit

### GUI Version
All console controls plus:
- **Mouse**: Not currently used (future custom waveform drawing)

## Dependencies

- `kivy` - Modern GUI framework for the interface
- `numpy` - Mathematical operations and array handling
- `keyboard` - Keyboard input handling (console version)
- `sounddevice` - Professional low-latency audio

## Audio Backends

### Professional Audio (Recommended)
- Uses `sounddevice` for low-latency, click-free audio
- Professional-grade audio quality with optimal latency settings
- Real-time parameter updates without audio artifacts

### Simple Audio (Fallback)
- Uses `pygame` mixer for basic audio functionality
- Good for systems without sounddevice support
- May have slightly higher latency but fully functional
- Used by console version

## Features by Version

### GUI Version (wave_generator.py)
- Modern graphical user interface using Kivy
- Real-time waveform visualization with animation
- Professional audio backend using sounddevice
- Smooth parameter transitions and anti-click protection

### Console Version (wave_generator_console.py)
- Text-based menu interface
- Basic wave generation and playback using pygame
- Frequency sweep functionality
- Keyboard-based controls
- Lightweight with minimal dependencies

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by submitting pull requests or reporting issues.
