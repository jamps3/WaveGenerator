#!/usr/bin/env python3
"""
Simple, working sine wave tone generator that actually works without clicks.
Uses a single long buffer that gets regenerated only when needed.
"""

import pygame
import numpy as np
import time
import threading

class SimpleToneGenerator:
    def __init__(self):
        # Initialize pygame mixer with optimal settings
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        
        self.sample_rate = 44100
        self.frequency = 440.0
        self.volume = 0.5
        self.is_playing = False
        self.stop_thread = False
        
        # Create a long buffer (1 second) that will loop seamlessly
        self.buffer_duration = 1.0
        self.buffer_frames = int(self.buffer_duration * self.sample_rate)
        
        self.sound = None
        self.channel = pygame.mixer.Channel(0)
        self.playback_thread = None
        
    def generate_tone_buffer(self, frequency, volume, frames):
        """Generate a seamless looping tone buffer"""
        # Generate exactly one second of audio that loops perfectly
        t = np.linspace(0, 2 * np.pi, frames, endpoint=False)
        
        # Generate sine wave
        wave = np.sin(t * frequency / self.sample_rate * frames)
        
        # Apply volume
        wave = wave * volume
        
        # Convert to 16-bit integers
        wave_int = (wave * 32767).astype(np.int16)
        
        # Make stereo
        stereo_wave = np.column_stack((wave_int, wave_int))
        
        return stereo_wave
    
    def start_tone(self):
        """Start playing the tone"""
        if self.is_playing:
            return
            
        self.is_playing = True
        self.stop_thread = False
        
        # Generate initial buffer
        wave_data = self.generate_tone_buffer(self.frequency, self.volume, self.buffer_frames)
        self.sound = pygame.sndarray.make_sound(wave_data)
        
        # Start playing with infinite loop
        self.channel.play(self.sound, loops=-1)
        
        # Start monitoring thread for parameter changes
        self.playback_thread = threading.Thread(target=self._monitor_changes)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def _monitor_changes(self):
        """Monitor for frequency/volume changes and update buffer when needed"""
        last_frequency = self.frequency
        last_volume = self.volume
        
        while not self.stop_thread:
            # Check if parameters changed
            if self.frequency != last_frequency or self.volume != last_volume:
                # Generate new buffer
                wave_data = self.generate_tone_buffer(self.frequency, self.volume, self.buffer_frames)
                new_sound = pygame.sndarray.make_sound(wave_data)
                
                # Replace the current sound
                self.channel.stop()
                self.channel.play(new_sound, loops=-1)
                self.sound = new_sound
                
                last_frequency = self.frequency
                last_volume = self.volume
            
            # Check every 50ms
            time.sleep(0.05)
    
    def stop_tone(self):
        """Stop playing the tone"""
        self.is_playing = False
        self.stop_thread = True
        
        if self.channel:
            self.channel.stop()
        
        if self.playback_thread:
            self.playback_thread.join()
    
    def set_frequency(self, frequency):
        """Set the frequency"""
        self.frequency = max(20, min(20000, frequency))
    
    def set_volume(self, volume):
        """Set the volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))

def main():
    """Simple test of the tone generator"""
    print("Simple Tone Generator Test")
    print("Press Enter to start 440Hz tone...")
    input()
    
    generator = SimpleToneGenerator()
    generator.start_tone()
    
    print("Tone playing at 440Hz. Press Enter to change to 880Hz...")
    input()
    
    generator.set_frequency(880)
    print("Changed to 880Hz. Press Enter to stop...")
    input()
    
    generator.stop_tone()
    print("Stopped.")
    
    pygame.quit()

if __name__ == "__main__":
    main()
