# Wave Generator

A Python-based wave generator with both console and GUI interfaces, featuring custom waveform drawing capabilities.

## Features

- **Multiple Wave Types**: Sine, square, sawtooth, and custom waveforms
- **Frequency Control**: Adjustable frequency with customizable step sizes
- **Volume Control**: 0-100% volume adjustment
- **Frequency Sweep**: Sweep through frequency ranges with customizable duration and looping
- **Custom Waveform Drawing**: Draw your own waveforms using mouse input (GUI version)
- **Real-time Visualization**: Live waveform display with animation
- **Dual Interfaces**: Choose between console-based or GUI-based operation

## Files

- `wave_generator_console.py` - Original console-based interface
- `wave_generator_gui.py` - Enhanced GUI interface with custom waveform drawing
- `requirements.txt` - Python dependencies
- `key_test.py` - Keyboard input testing utility

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Console Version
```bash
python wave_generator_console.py
```

### GUI Version
```bash
python wave_generator_gui.py
```

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

## Features by Version

### Console Version
- Text-based menu interface
- Basic wave generation and playback
- Frequency sweep functionality
- Keyboard-based controls

### GUI Version
- Graphical user interface with icons
- Real-time waveform visualization
- Custom waveform drawing with mouse
- Enhanced sweep visualization
- Visual feedback for all controls

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by submitting pull requests or reporting issues.
