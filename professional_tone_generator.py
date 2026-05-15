#!/usr/bin/env python3
"""
Professional tone generator using sounddevice for click-free, low-latency audio.
This is the proper way to do real-time audio generation.
"""

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not found. Install with: pip install sounddevice")
    exit(1)

import numpy as np
import threading
import time

class ProfessionalToneGenerator:
    def __init__(self, sample_rate=44100, block_size=512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Audio parameters
        self.frequency = 440.0
        self.volume = 0.3
        self.is_playing = False
        
        # Phase accumulator for perfect continuity
        self.phase = 0.0
        self.phase_increment = 0.0
        
        # Thread-safe parameter updates
        self.param_lock = threading.Lock()
        
        # Audio stream
        self.stream = None
        
    def audio_callback(self, outdata, frames, time, status):
        """Audio callback - called by sounddevice for each audio block"""
        if status:
            print(f"Audio status: {status}")
        
        with self.param_lock:
            # Update phase increment if frequency changed
            self.phase_increment = 2 * np.pi * self.frequency / self.sample_rate
            
            # Generate audio samples
            phases = self.phase + np.arange(frames) * self.phase_increment
            samples = np.sin(phases) * self.volume
            
            # Update phase for next callback (maintain continuity)
            self.phase = (self.phase + frames * self.phase_increment) % (2 * np.pi)
        
        # Output stereo
        outdata[:, 0] = samples  # Left
        outdata[:, 1] = samples  # Right
    
    def start(self):
        """Start audio playback"""
        if self.is_playing:
            return
        
        self.is_playing = True
        self.phase = 0.0
        
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.block_size,
                callback=self.audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            print(f"Audio started: {self.sample_rate}Hz, block size {self.block_size}")
        except Exception as e:
            print(f"Error starting audio: {e}")
            self.is_playing = False
    
    def stop(self):
        """Stop audio playback"""
        self.is_playing = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def set_frequency(self, frequency):
        """Set frequency (thread-safe)"""
        with self.param_lock:
            self.frequency = max(20, min(20000, frequency))
    
    def set_volume(self, volume):
        """Set volume 0.0 to 1.0 (thread-safe)"""
        with self.param_lock:
            self.volume = max(0.0, min(1.0, volume))

def main():
    """Test the professional tone generator"""
    print("Professional Tone Generator")
    print("This uses sounddevice for professional, click-free audio")
    print()
    
    # Check available audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print()
    
    generator = ProfessionalToneGenerator()
    
    try:
        print("Starting 440Hz tone...")
        generator.start()
        time.sleep(2)
        
        print("Changing to 880Hz...")
        generator.set_frequency(880)
        time.sleep(2)
        
        print("Changing to 220Hz...")
        generator.set_frequency(220)
        time.sleep(2)
        
        print("Volume sweep...")
        for v in np.linspace(0.3, 0.05, 20):
            generator.set_volume(v)
            time.sleep(0.1)
        
        for v in np.linspace(0.05, 0.3, 20):
            generator.set_volume(v)
            time.sleep(0.1)
        
        print("Frequency sweep...")
        for f in np.linspace(220, 880, 100):
            generator.set_frequency(f)
            time.sleep(0.02)
        
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        print("Stopping...")
        generator.stop()
        print("Done.")

if __name__ == "__main__":
    main()
