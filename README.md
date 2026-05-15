# Wave Generator

A Python-based wave generator with modular audio backends, featuring professional-quality audio output, custom waveform drawing, and real-time visualization. Includes both GUI and console interfaces for maximum flexibility.

## Features

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

- `wave_generator.py` - Main GUI interface with custom waveform drawing and professional audio
- `wave_generator_console.py` - Console-based interface for text-based operation
- `wave_engine.py` - Shared wave generation engine with smooth transitions
- `professional_tone_generator.py` - High-quality audio generator using sounddevice
- `simple_tone_generator.py` - Basic audio generator using pygame
- `requirements.txt` - Python dependencies
- `key_test.py` - Keyboard input testing utility

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) For professional audio quality, install sounddevice:
   ```bash
   pip install sounddevice
   ```

## Usage

### GUI Version (Recommended)
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
- **C**: Toggle custom waveform drawing mode (when custom wave type is selected)
- **R**: Reset custom waveform to default sine wave
- **Mouse**: Draw custom waveforms when in drawing mode

## Custom Waveform Drawing

1. Select "Custom" as the wave type using arrow keys
2. Press 'C' to enter drawing mode
3. Click and drag in the waveform visualization area to draw your custom waveform
4. Press 'C' again to exit drawing mode
5. Press 'R' to reset the custom waveform

## Dependencies

- `pygame` - Audio playback and GUI
- `numpy` - Mathematical operations and array handling
- `keyboard` - Keyboard input handling (console version)
- `sounddevice` - Professional low-latency audio (optional, for enhanced audio quality)

## Audio Backends

### Professional Audio (Recommended)
- Uses `sounddevice` for low-latency, click-free audio
- Professional-grade audio quality with optimal latency settings
- Real-time parameter updates without audio artifacts

### Simple Audio (Fallback)
- Uses `pygame` mixer for basic audio functionality
- Good for systems without sounddevice support
- May have slightly higher latency but fully functional

## Features by Version

### GUI Version (wave_generator.py)
- Graphical user interface with icons and visual feedback
- Real-time waveform visualization with animation
- Custom waveform drawing with mouse input
- Enhanced frequency sweep visualization
- Professional audio backend (sounddevice) with fallback to pygame
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
