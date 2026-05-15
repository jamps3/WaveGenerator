import pygame
import numpy as np
import keyboard
import threading
import time
import os
import sys
import math
from wave_engine import WaveEngine

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not found. Install with: pip install sounddevice")
    print("This is required for professional click-free audio.")
    exit(1)


class WaveGenerator:
    def __init__(self):
        # Initialize pygame for GUI only (no audio)
        pygame.init()
        
        # Professional audio settings
        self.audio_sample_rate = 44100
        self.audio_block_size = 512
        
        # Audio state for sounddevice
        self.audio_stream = None
        self.audio_phase = 0.0
        self.audio_lock = threading.Lock()
        self.audio_running = False

        # Initialize graphical display
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 600
        self.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), vsync=1
        )
        pygame.display.set_caption("🎵 Wave Generator 🎵")

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (100, 149, 237)
        self.GREEN = (60, 179, 113)
        self.RED = (220, 20, 60)
        self.YELLOW = (255, 215, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (211, 211, 211)
        self.DARK_GRAY = (64, 64, 64)
        self.ORANGE = (255, 165, 0)

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)

        # Animation variables
        self.wave_animation_offset = 0
        self.last_frame_time = time.time()

        # Menu layout
        self.menu_start_y = 120
        self.menu_item_height = 40
        self.menu_padding = 20

        self.duration = 10.0  # Duration of each wave cycle in seconds 
        self.is_playing = False
        
        # Initialize the shared wave engine with audio sample rate
        self.wave_engine = WaveEngine(sample_rate=self.audio_sample_rate)

        # Menu options
        self.menu_options = [
            "Tone Playback",
            "Volume (%)",
            "Hz Step",
            "Wave Type",
            "Frequency (Hz)",
            "Sweep",
            "Sweep Duration",
            "Sweep Start",
            "Sweep End",
            "Sweep Loop",
        ]
        self.current_option = 0

        # Settings
        self.frequency = 440.0  # Default A4 note
        self.volume = 50  # 50% volume
        self.wave_type = "sine"  # Default wave type
        self.wave_types = ["sine", "square", "sawtooth", "custom"]

        # Custom waveform data
        self.custom_waveform = np.sin(
            np.linspace(0, 2 * np.pi, 1000)
        )  # Default to sine wave
        self.drawing_mode = False
        self.drawing_points = []

        # Hz step options
        self.hz_steps = [0.1, 1.0, 10.0, 100.0, 1000.0]
        self.current_hz_step_index = 2  # Start with 10Hz steps (reasonable default)
        self.current_hz_step = self.hz_steps[self.current_hz_step_index]

        # Frequency range limits
        self.min_frequency = 0.1
        self.max_frequency = 20000.0

        # Sweep settings
        self.sweep_duration = 1.0  # default 1 second
        self.sweep_start = 200.0  # default start frequency
        self.sweep_end = 20000.0  # default end frequency (20kHz)
        self.sweep_active = False
        self.sweep_thread = None
        self.sweep_durations = [
            0.01,
            0.1,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
        ]  # Available durations
        self.current_sweep_duration_index = 2  # default to 1 second
        self.sweep_current_frequency = 0.0  # Current frequency during sweep
        self.user_selected_frequency = (
            440.0  # User's selected frequency (preserved during sweep)
        )
        self.sweep_loop = True  # Whether sweep should loop continuously (default on)
        
        # Initialize audio system
        self._initialize_audio()

    def _initialize_audio(self):
        """Initialize the professional audio system with sounddevice"""
        try:
            # Query available audio devices for optimal setup
            devices = sd.query_devices()
            default_device = sd.default.device
            print(f"Using audio device: {devices[default_device[1]]['name']}")
            
            # Set optimal latency settings
            sd.default.samplerate = self.audio_sample_rate
            sd.default.channels = 2  # Stereo output
            sd.default.dtype = 'float32'
            sd.default.latency = 'low'  # Low latency for real-time audio
            
        except Exception as e:
            print(f"Warning: Could not initialize optimal audio settings: {e}")
            print("Using default audio configuration.")
    
    def _audio_callback(self, outdata, frames, time, status):
        """Professional audio callback for sounddevice - generates audio in real-time"""
        if status:
            print(f"Audio callback status: {status}")
        
        with self.audio_lock:
            if not self.audio_running:
                outdata.fill(0)
                return
            
            # Generate audio samples for current parameters
            current_freq = self.frequency
            current_vol = self.volume / 100.0
            current_wave = self.wave_type
            
            # Handle sweep mode
            if self.sweep_active and self.sweep_current_frequency > 0:
                current_freq = self.sweep_current_frequency
            
            # Generate samples
            for i in range(frames):
                # Calculate the current phase
                phase_increment = 2.0 * np.pi * current_freq / self.audio_sample_rate
                
                # Generate waveform sample
                if current_wave == "sine":
                    sample = np.sin(self.audio_phase)
                elif current_wave == "square":
                    sample = 1.0 if np.sin(self.audio_phase) >= 0 else -1.0
                elif current_wave == "sawtooth":
                    normalized_phase = self.audio_phase % (2 * np.pi)
                    sample = 2.0 * (normalized_phase / (2 * np.pi)) - 1.0
                elif current_wave == "custom":
                    normalized_phase = self.audio_phase % (2 * np.pi)
                    index = int(normalized_phase / (2 * np.pi) * (len(self.custom_waveform) - 1))
                    index = max(0, min(index, len(self.custom_waveform) - 1))
                    sample = self.custom_waveform[index]
                else:
                    sample = np.sin(self.audio_phase)  # Default to sine
                
                # Apply volume
                sample *= current_vol
                
                # Output to both stereo channels
                outdata[i] = [sample, sample]
                
                # Update phase for next sample
                self.audio_phase += phase_increment
                
                # Keep phase in reasonable range to prevent overflow
                if self.audio_phase >= 2 * np.pi:
                    self.audio_phase -= 2 * np.pi

    def start_audio_stream(self):
        """Start the professional audio stream"""
        try:
            if self.audio_stream is not None:
                self.stop_audio_stream()
            
            with self.audio_lock:
                self.audio_running = True
                self.audio_phase = 0.0
            
            # Create and start the audio stream
            self.audio_stream = sd.OutputStream(
                channels=2,
                samplerate=self.audio_sample_rate,
                callback=self._audio_callback,
                blocksize=self.audio_block_size,
                dtype='float32',
                latency='low'
            )
            
            self.audio_stream.start()
            print(f"Audio stream started: {self.audio_sample_rate}Hz, {self.audio_block_size} samples")
            
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            with self.audio_lock:
                self.audio_running = False
    
    def stop_audio_stream(self):
        """Stop the professional audio stream"""
        try:
            with self.audio_lock:
                self.audio_running = False
            
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                print("Audio stream stopped")
                
        except Exception as e:
            print(f"Error stopping audio stream: {e}")
    
    def create_anti_click_transition(self, old_wave_type, old_frequency, old_volume,
                                   new_wave_type, new_frequency, new_volume,
                                   start_phase, crossfade_frames, loop_frames):
        """Create an anti-click transition that prevents snapping sounds during frequency changes"""
        total_frames = crossfade_frames + loop_frames
        arr = np.zeros((total_frames, 2), dtype=np.int16)
        
        # Calculate current amplitude at the transition point for seamless matching
        old_amplitude = self._get_wave_amplitude_at_phase(old_wave_type, start_phase, old_volume)
        
        # Generate the transition
        for i in range(total_frames):
            if i < crossfade_frames:
                # Crossfade region - blend between old and new waveforms
                progress = i / crossfade_frames
                
                # Calculate phases
                old_phase = start_phase + (i / self.sample_rate) * old_frequency * 2 * np.pi
                new_phase = start_phase + (i / self.sample_rate) * new_frequency * 2 * np.pi
                
                # Get waveform values
                old_value = self._get_wave_value_at_phase(old_wave_type, old_phase) * (old_volume / 100.0)
                new_value = self._get_wave_value_at_phase(new_wave_type, new_phase) * (new_volume / 100.0)
                
                # Smooth crossfade with cosine interpolation to prevent clicks
                fade_factor = 0.5 * (1 - np.cos(progress * np.pi))
                wave_value = old_value * (1 - fade_factor) + new_value * fade_factor
            else:
                # Pure new waveform after crossfade
                sample_index = i - crossfade_frames
                phase = start_phase + ((crossfade_frames + sample_index) / self.sample_rate) * new_frequency * 2 * np.pi
                wave_value = self._get_wave_value_at_phase(new_wave_type, phase) * (new_volume / 100.0)
            
            # Convert to 16-bit integer with proper clipping
            wave_value_int = int(np.clip(wave_value * 32767, -32767, 32767))
            arr[i, 0] = wave_value_int
            arr[i, 1] = wave_value_int
        
        return arr
    
    def _get_wave_amplitude_at_phase(self, wave_type, phase, volume):
        """Get the amplitude of a waveform at a specific phase"""
        amplitude = self._get_wave_value_at_phase(wave_type, phase)
        return amplitude * (volume / 100.0)
    
    def _get_wave_value_at_phase(self, wave_type, phase):
        """Get the normalized waveform value at a specific phase"""
        if wave_type == "sine":
            return np.sin(phase)
        elif wave_type == "square":
            return np.sign(np.sin(phase))
        elif wave_type == "sawtooth":
            normalized_phase = phase % (2 * np.pi)
            return 2 * (normalized_phase / (2 * np.pi)) - 1
        elif wave_type == "custom":
            normalized_phase = phase % (2 * np.pi)
            index = int(normalized_phase / (2 * np.pi) * (len(self.custom_waveform) - 1))
            # Ensure index is within bounds
            index = max(0, min(index, len(self.custom_waveform) - 1))
            return self.custom_waveform[index]
        else:
            return np.sin(phase)

    def sweep_frequency(self, sweep_duration=None, sweep_start=None, sweep_end=None):
        """Sweep frequency over a specified range and duration with smooth transitions"""
        sweep_duration = sweep_duration or self.sweep_duration
        sweep_start = sweep_start or self.sweep_start
        sweep_end = sweep_end or self.sweep_end

        self.user_selected_frequency = self.frequency

        # Ultra-high resolution for maximum smoothness - 1000 updates per second
        # Ensure minimum number of steps to avoid division by zero
        steps = max(1, int(sweep_duration * 1000))
        step_duration = sweep_duration / steps
        frequency_step = (sweep_end - sweep_start) / steps

        while True:
            start_time = time.time()
            for i in range(steps):
                if not self.sweep_active:
                    break

                # Calculate precise frequency with exponential sweep option for more natural feel
                # Use linear sweep for simplicity, but with much higher resolution
                progress = i / steps
                self.sweep_current_frequency = (
                    sweep_start + (sweep_end - sweep_start) * progress
                )

                # Use precise timing to maintain smooth sweep
                target_time = start_time + i * step_duration
                current_time = time.time()
                sleep_time = target_time - current_time

                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not self.sweep_loop or not self.sweep_active:
                break

        self.sweep_active = False
        self.sweep_current_frequency = 0.0


    def _stop_sweep_threads(self):
        """Helper method to stop sweep threads safely"""
        if self.sweep_thread and self.sweep_thread.is_alive():
            self.sweep_thread.join()

    def _start_sweep_threads(self):
        """Helper method to start sweep threads"""
        self.sweep_thread = threading.Thread(target=self.sweep_frequency)
        self.sweep_thread.daemon = True
        self.sweep_thread.start()

    def _restart_sweep_if_active(self):
        """Helper method to restart sweep if currently active"""
        if self.sweep_active:
            self.sweep_active = False
            self._stop_sweep_threads()
            self.sweep_active = True
            self._start_sweep_threads()

    def toggle_sweep(self):
        """Toggle frequency sweep on/off"""
        if self.sweep_active:
            self.sweep_active = False
            self._stop_sweep_threads()
        else:
            self.sweep_active = True
            self._start_sweep_threads()

    def adjust_sweep_duration(self, direction):
        """Adjust sweep duration up or down"""
        if direction == "up":
            self.current_sweep_duration_index = (
                self.current_sweep_duration_index + 1
            ) % len(self.sweep_durations)
        elif direction == "down":
            self.current_sweep_duration_index = (
                self.current_sweep_duration_index - 1
            ) % len(self.sweep_durations)
        self.sweep_duration = self.sweep_durations[self.current_sweep_duration_index]

        # If sweep is currently active, restart it with the new duration
        self._restart_sweep_if_active()

    def adjust_sweep_start(self, direction):
        """Adjust sweep start frequency up or down"""
        if direction == "up":
            self.sweep_start = min(
                self.sweep_start + self.current_hz_step, self.sweep_end - 1.0
            )
        elif direction == "down":
            self.sweep_start = max(
                self.sweep_start - self.current_hz_step, self.min_frequency
            )

        # Round to appropriate decimal places
        if self.current_hz_step >= 1.0:
            self.sweep_start = round(self.sweep_start, 1)
        else:
            self.sweep_start = round(self.sweep_start, 1)

        # If sweep is currently active, restart it with the new start frequency
        self._restart_sweep_if_active()

    def adjust_sweep_end(self, direction):
        """Adjust sweep end frequency up or down"""
        if direction == "up":
            self.sweep_end = min(
                self.sweep_end + self.current_hz_step, self.max_frequency
            )
        elif direction == "down":
            self.sweep_end = max(
                self.sweep_end - self.current_hz_step, self.sweep_start + 1.0
            )

        # Round to appropriate decimal places
        if self.current_hz_step >= 1.0:
            self.sweep_end = round(self.sweep_end, 1)
        else:
            self.sweep_end = round(self.sweep_end, 1)

        # If sweep is currently active, restart it with the new end frequency
        self._restart_sweep_if_active()

    def play_continuous_sound(self):
        """Simple continuous tone generator with real-time parameter updates"""
        # Simple approach: small chunks played continuously with overlap
        chunk_duration = 0.05  # 50ms chunks - responsive but stable
        chunk_frames = int(chunk_duration * self.sample_rate)
        overlap_frames = chunk_frames // 4  # 25% overlap to prevent gaps
        
        # Initialize playback state
        current_phase = 0.0
        channels = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]
        current_channel = 0
        
        for channel in channels:
            channel.set_volume(1.0)
        
        while not self.stop_sound:
            try:
                # Generate chunk with current parameters
                wave_data = self.generate_wave_chunk(
                    self.wave_type, self.frequency, self.volume, chunk_frames, current_phase
                )
                
                # Update phase for next chunk
                phase_increment = (chunk_frames / self.sample_rate) * self.frequency * 2 * np.pi
                current_phase = (current_phase + phase_increment) % (2 * np.pi)
                
                # Play chunk on alternating channels
                sound = pygame.sndarray.make_sound(wave_data)
                channels[current_channel].play(sound)
                current_channel = (current_channel + 1) % 2
                
                # Wait for most of the chunk to play, but start next one early for overlap
                time.sleep(chunk_duration - (overlap_frames / self.sample_rate))
                
            except Exception as e:
                print(f"Error in playback: {e}")
                time.sleep(0.01)
        
        # Stop all channels
        for channel in channels:
            channel.stop()
    
    def _find_matching_phase(self, wave_type, target_amplitude, volume, current_phase):
        """Find the phase that produces the target amplitude for seamless transitions"""
        if volume == 0:
            return current_phase
        
        # Normalize target amplitude to wave range (-1 to 1)
        normalized_target = target_amplitude / (volume / 100.0)
        normalized_target = np.clip(normalized_target, -1.0, 1.0)
        
        if wave_type == "sine":
            # For sine wave, find phase that gives target amplitude
            try:
                # Get the primary solution
                phase_candidate = np.arcsin(normalized_target)
                
                # There are multiple solutions - choose the one closest to current phase
                candidates = [
                    phase_candidate,
                    np.pi - phase_candidate,
                    phase_candidate + 2 * np.pi,
                    np.pi - phase_candidate + 2 * np.pi,
                    phase_candidate - 2 * np.pi,
                    np.pi - phase_candidate - 2 * np.pi
                ]
                
                # Choose the candidate closest to current phase
                best_phase = min(candidates, key=lambda x: abs((x - current_phase + np.pi) % (2 * np.pi) - np.pi))
                return best_phase % (2 * np.pi)
                
            except (ValueError, OverflowError):
                return current_phase
                
        elif wave_type == "square":
            # For square wave, match the sign
            if normalized_target >= 0:
                # Positive region: phase in [0, π)
                target_phase = np.pi / 2  # Middle of positive region
            else:
                # Negative region: phase in [π, 2π)
                target_phase = 3 * np.pi / 2  # Middle of negative region
            return target_phase
            
        elif wave_type == "sawtooth":
            # For sawtooth: value = 2 * (phase / (2π)) - 1
            # Solve for phase: phase = π * (normalized_target + 1)
            target_phase = np.pi * (normalized_target + 1)
            return target_phase % (2 * np.pi)
            
        elif wave_type == "custom":
            # For custom waveform, find the closest matching value
            if len(self.custom_waveform) == 0:
                return current_phase
                
            # Find the index with the closest value
            differences = np.abs(self.custom_waveform - normalized_target)
            min_index = np.argmin(differences)
            
            # Convert index back to phase
            target_phase = (min_index / len(self.custom_waveform)) * 2 * np.pi
            return target_phase
            
        else:
            # Default to current phase
            return current_phase

    def find_nearest_zero_crossing(self, current_phase, wave_type):
        """Find the nearest zero-crossing point for smooth waveform transitions"""
        # Normalize phase to [0, 2π]
        phase = current_phase % (2 * np.pi)

        if wave_type == "sine":
            # For sine wave, zero crossings are at 0, π, 2π
            zero_crossings = [0, np.pi, 2 * np.pi]
        elif wave_type == "square":
            # For square wave, zero crossings are at 0, π, 2π (transitions)
            zero_crossings = [0, np.pi, 2 * np.pi]
        elif wave_type == "sawtooth":
            # For sawtooth wave, zero crossing is at π (middle of the cycle)
            zero_crossings = [0, np.pi, 2 * np.pi]
        elif wave_type == "custom":
            # For custom waveform, find actual zero crossings
            zero_crossings = self.find_custom_zero_crossings()
        else:
            # Default to sine wave zero crossings
            zero_crossings = [0, np.pi, 2 * np.pi]

        # Find the nearest zero crossing
        nearest_crossing = min(
            zero_crossings,
            key=lambda x: min(
                abs(x - phase), abs(x - phase + 2 * np.pi), abs(x - phase - 2 * np.pi)
            ),
        )

        return nearest_crossing

    def find_custom_zero_crossings(self):
        """Find zero crossings in custom waveform"""
        zero_crossings = [0, 2 * np.pi]  # Always include start and end

        # Find actual zero crossings in the custom waveform
        for i in range(len(self.custom_waveform) - 1):
            if (self.custom_waveform[i] <= 0 and self.custom_waveform[i + 1] > 0) or (
                self.custom_waveform[i] >= 0 and self.custom_waveform[i + 1] < 0
            ):
                # Zero crossing found, convert index to phase
                phase = (i / len(self.custom_waveform)) * 2 * np.pi
                zero_crossings.append(phase)

        return sorted(zero_crossings)

    def generate_fast_transition_waveform(
        self,
        old_wave_type,
        old_frequency,
        old_volume,
        new_wave_type,
        new_frequency,
        new_volume,
        start_phase,
        transition_frames,
        loop_frames,
    ):
        """Generate a waveform with ultra-fast transition between frequencies"""
        # Sync custom waveform with wave engine
        self.wave_engine.set_custom_waveform(self.custom_waveform)
        return self.wave_engine.generate_fast_transition_waveform(
            old_wave_type, old_frequency, old_volume,
            new_wave_type, new_frequency, new_volume,
            start_phase, transition_frames, loop_frames
        )

    def toggle_play_pause(self):
        """Toggle play/pause state using professional sounddevice audio"""
        if self.is_playing:
            # Stop the audio stream
            self.stop_audio_stream()
            self.is_playing = False
        else:
            # Start the audio stream
            self.start_audio_stream()
            self.is_playing = True

    def adjust_frequency(self, direction):
        """Adjust frequency up or down using current Hz step"""
        # If sweep is active, adjust the user's selected frequency (preserved)
        # Otherwise, adjust the current frequency
        if self.sweep_active:
            # Adjust user's selected frequency (what they'll get when sweep ends)
            if direction == "up":
                self.user_selected_frequency = min(
                    self.user_selected_frequency + self.current_hz_step,
                    self.max_frequency,
                )
            elif direction == "down":
                self.user_selected_frequency = max(
                    self.user_selected_frequency - self.current_hz_step,
                    self.min_frequency,
                )

            # Round to appropriate decimal places based on step size
            if self.current_hz_step >= 1.0:
                self.user_selected_frequency = round(self.user_selected_frequency, 1)
            else:
                self.user_selected_frequency = round(self.user_selected_frequency, 1)
        else:
            # Normal frequency adjustment when sweep is not active
            if direction == "up":
                self.frequency = min(
                    self.frequency + self.current_hz_step, self.max_frequency
                )
            elif direction == "down":
                self.frequency = max(
                    self.frequency - self.current_hz_step, self.min_frequency
                )

            # Round to appropriate decimal places based on step size
            if self.current_hz_step >= 1.0:
                self.frequency = round(self.frequency, 1)
            else:
                self.frequency = round(self.frequency, 1)

            # Update user selected frequency to match
            self.user_selected_frequency = self.frequency

    def adjust_volume(self, direction):
        """Adjust volume up or down"""
        if direction == "up":
            self.volume = min(self.volume + 5, 100)  # Max 100%
        elif direction == "down":
            self.volume = max(self.volume - 5, 0)  # Min 0%

    def cycle_wave_type(self, direction):
        """Cycle through wave types"""
        current_index = self.wave_types.index(self.wave_type)
        if direction == "up":
            current_index = (current_index + 1) % len(self.wave_types)
        elif direction == "down":
            current_index = (current_index - 1) % len(self.wave_types)
        self.wave_type = self.wave_types[current_index]

    def cycle_hz_step(self, direction):
        """Cycle through Hz step options"""
        if direction == "up":
            self.current_hz_step_index = (self.current_hz_step_index + 1) % len(
                self.hz_steps
            )
        elif direction == "down":
            self.current_hz_step_index = (self.current_hz_step_index - 1) % len(
                self.hz_steps
            )
        self.current_hz_step = self.hz_steps[self.current_hz_step_index]

    def clear_screen(self):
        """Clear the console screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def draw_play_icon(self, x, y, size=16, color=None):
        """Draw a simple play triangle icon"""
        if color is None:
            color = self.GREEN
        points = [(x, y), (x, y + size), (x + size, y + size // 2)]
        pygame.draw.polygon(self.screen, color, points)

    def draw_pause_icon(self, x, y, size=16, color=None):
        """Draw a simple pause icon (two rectangles)"""
        if color is None:
            color = self.RED
        bar_width = size // 3
        bar_height = size
        # Left bar
        pygame.draw.rect(self.screen, color, (x, y, bar_width, bar_height))
        # Right bar
        pygame.draw.rect(
            self.screen, color, (x + bar_width + 2, y, bar_width, bar_height)
        )

    def draw_speaker_icon(self, x, y, size=16, color=None):
        """Draw a simple speaker icon"""
        if color is None:
            color = self.GREEN
        # Speaker base
        pygame.draw.rect(self.screen, color, (x, y + size // 3, size // 2, size // 3))
        # Speaker cone
        points = [
            (x + size // 2, y + size // 4),
            (x + size // 2, y + 3 * size // 4),
            (x + size, y + size // 8),
            (x + size, y + 7 * size // 8),
        ]
        pygame.draw.polygon(self.screen, color, points)

    def draw_wave_icon(self, x, y, size=16, color=None):
        """Draw a simple wave icon"""
        if color is None:
            color = self.BLUE
        # Draw a simple sine wave
        points = []
        for i in range(size):
            wave_y = y + size // 2 + int(math.sin(i * 0.5) * size // 4)
            points.append((x + i, wave_y))
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def draw_volume_bar(self, x, y, width, height, volume_percent):
        """Draw a graphical volume bar"""
        # Background
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x, y, width, height))
        # Fill
        fill_width = int(width * (volume_percent / 100.0))
        pygame.draw.rect(self.screen, self.GREEN, (x, y, fill_width, height))
        # Border
        pygame.draw.rect(self.screen, self.WHITE, (x, y, width, height), 1)

    def draw_circle_icon(self, x, y, size=8, color=None, filled=True):
        """Draw a simple circle icon"""
        if color is None:
            color = self.GREEN
        if filled:
            pygame.draw.circle(
                self.screen, color, (x + size // 2, y + size // 2), size // 2
            )
        else:
            pygame.draw.circle(
                self.screen, color, (x + size // 2, y + size // 2), size // 2, 2
            )

    def draw_refresh_icon(self, x, y, size=16, color=None):
        """Draw a simple refresh/loop icon (curved arrow)"""
        if color is None:
            color = self.BLUE
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 3

        # Draw arc
        pygame.draw.arc(
            self.screen,
            color,
            (center_x - radius, center_y - radius, radius * 2, radius * 2),
            0,
            math.pi * 1.5,
            2,
        )
        # Draw arrow
        arrow_points = [
            (center_x + radius - 2, center_y - radius),
            (center_x + radius + 2, center_y - radius - 3),
            (center_x + radius + 2, center_y - radius + 3),
        ]
        pygame.draw.polygon(self.screen, color, arrow_points)

    def draw_arrow_keys_icon(self, x, y, size=16, color=None):
        """Draw simple arrow keys icon"""
        if color is None:
            color = self.LIGHT_GRAY
        arrow_up = [
            (x + size / 2, y),
            (x + size / 3, y + size / 3),
            (x + 2 * size / 3, y + size / 3),
        ]
        arrow_down = [
            (x + size / 2, y + size),
            (x + size / 3, y + 2 * size / 3),
            (x + 2 * size / 3, y + 2 * size / 3),
        ]
        arrow_left = [
            (x, y + size / 2),
            (x + size / 3, y + size / 3),
            (x + size / 3, y + 2 * size / 3),
        ]
        arrow_right = [
            (x + size, y + size / 2),
            (x + 2 * size / 3, y + size / 3),
            (x + 2 * size / 3, y + 2 * size / 3),
        ]
        pygame.draw.polygon(self.screen, color, arrow_up)
        pygame.draw.polygon(self.screen, color, arrow_down)
        pygame.draw.polygon(self.screen, color, arrow_left)
        pygame.draw.polygon(self.screen, color, arrow_right)

    def draw_space_key_icon(self, x, y, width=30, height=16, color=None):
        """Draw a simple space key icon"""
        if color is None:
            color = self.LIGHT_GRAY
        pygame.draw.rect(self.screen, color, (x, y, width, height))
        pygame.draw.rect(self.screen, self.BLACK, (x + 2, y + 2, width - 4, height - 4))
        pygame.draw.rect(self.screen, color, (x, y, width, height), 3)

    def draw_q_key_icon(self, x, y, size=16, color=None):
        """Draw a simple Q key icon"""
        if color is None:
            color = self.LIGHT_GRAY
        pygame.draw.rect(self.screen, color, (x, y, size, size))
        q_surface = self.font_small.render("Q", True, self.BLACK)
        self.screen.blit(q_surface, (x + size // 4, y + size // 8))

    def display_menu(self):
        """Display the current menu state visually"""
        self.screen.fill(self.BLACK)  # Clear the screen with black

        # Draw title with wave icons
        title_text = "Wave Generator"
        title_surface = self.font_large.render(title_text, True, self.YELLOW)
        title_x = self.WINDOW_WIDTH / 2 - title_surface.get_width() / 2
        self.screen.blit(title_surface, (title_x, 20))

        # Draw wave icons on both sides of title
        self.draw_wave_icon(title_x - 30, 25, 20, self.YELLOW)
        self.draw_wave_icon(
            title_x + title_surface.get_width() + 10, 25, 20, self.YELLOW
        )

        # Draw status with icon
        is_anything_playing = self.is_playing or self.sweep_active
        status_text = "Playing" if is_anything_playing else "Paused"
        status_surface = self.font_medium.render(
            status_text, True, self.GREEN if is_anything_playing else self.RED
        )
        status_x = self.WINDOW_WIDTH / 2 - status_surface.get_width() / 2
        self.screen.blit(status_surface, (status_x, 70))

        # Draw play/pause icon next to status
        if is_anything_playing:
            self.draw_speaker_icon(status_x - 25, 72, 16, self.GREEN)
        else:
            self.draw_pause_icon(status_x - 25, 72, 16, self.RED)

        # Draw menu options
        for i, option in enumerate(self.menu_options):
            menu_x = self.menu_padding
            menu_y = self.menu_start_y + i * self.menu_item_height

            if i == 0:  # Play/Pause
                status = "Playing" if self.is_playing else "Paused"
                menu_text = f"{option}: {status}"

                # Draw play/pause icon
                if self.is_playing:
                    self.draw_speaker_icon(menu_x - 20, menu_y + 2, 12, self.GREEN)
                else:
                    self.draw_pause_icon(menu_x - 20, menu_y + 2, 12, self.RED)

            elif i == 1:  # Volume
                menu_text = f"{option}: {self.volume}%"

                # Draw volume bar after text
                text_surface = self.font_small.render(menu_text, True, self.WHITE)
                text_width = text_surface.get_width()
                self.draw_volume_bar(
                    menu_x + text_width + 10, menu_y + 8, 100, 8, self.volume
                )

            elif i == 2:  # Hz Step
                hz_step_displays = []
                for j, step in enumerate(self.hz_steps):
                    if step >= 1000:
                        step_display = f"{step/1000:.1f}kHz"
                    else:
                        step_display = f"{step:.1f}Hz"

                    if j == self.current_hz_step_index:
                        hz_step_displays.append(f"[{step_display}]")
                    else:
                        hz_step_displays.append(step_display)
                menu_text = f"{option}: {', '.join(hz_step_displays)}"

            elif i == 3:  # Wave Type
                wave_type_displays = []
                for wave_type in self.wave_types:
                    if wave_type == self.wave_type:
                        wave_type_displays.append(f"[{wave_type.capitalize()}]")
                    else:
                        wave_type_displays.append(wave_type.capitalize())
                menu_text = f"{option}: {', '.join(wave_type_displays)}"

                # Draw wave icon
                self.draw_wave_icon(menu_x - 20, menu_y + 2, 12, self.BLUE)

            elif i == 4:  # Frequency
                if self.frequency >= 1000:
                    freq_display = f"{self.frequency/1000:.1f}kHz"
                else:
                    freq_display = f"{self.frequency:.1f}Hz"
                menu_text = f"{option}: {freq_display}"

            elif i == 5:  # Sweep
                sweep_status = "Active" if self.sweep_active else "Inactive"
                if self.sweep_active and self.sweep_current_frequency > 0:
                    if self.sweep_current_frequency >= 1000:
                        current_display = f"{self.sweep_current_frequency/1000:.1f}kHz"
                    else:
                        current_display = f"{self.sweep_current_frequency:.1f}Hz"
                    menu_text = f"{option}: {sweep_status} (Current: {current_display})"
                else:
                    menu_text = f"{option}: {sweep_status}"

                # Draw sweep icon
                if self.sweep_active:
                    self.draw_refresh_icon(menu_x - 20, menu_y + 2, 12, self.GREEN)
                else:
                    self.draw_pause_icon(menu_x - 20, menu_y + 2, 12, self.RED)

            elif i == 6:  # Sweep Duration
                duration_displays = []
                for j, duration in enumerate(self.sweep_durations):
                    if j == self.current_sweep_duration_index:
                        duration_displays.append(f"[{duration}s]")
                    else:
                        duration_displays.append(f"{duration}s")
                menu_text = f"{option}: {', '.join(duration_displays)}"

            elif i == 7:  # Sweep Start
                if self.sweep_start >= 1000:
                    start_display = f"{self.sweep_start/1000:.1f}kHz"
                else:
                    start_display = f"{self.sweep_start:.1f}Hz"
                menu_text = f"{option}: {start_display}"

            elif i == 8:  # Sweep End
                if self.sweep_end >= 1000:
                    end_display = f"{self.sweep_end/1000:.1f}kHz"
                else:
                    end_display = f"{self.sweep_end:.1f}Hz"
                menu_text = f"{option}: {end_display}"

            elif i == 9:  # Sweep Loop
                loop_status = "On" if self.sweep_loop else "Off"
                menu_text = f"{option}: {loop_status}"

                # Draw loop icon
                if self.sweep_loop:
                    self.draw_refresh_icon(menu_x - 20, menu_y + 2, 12, self.GREEN)
                else:
                    self.draw_circle_icon(menu_x - 20, menu_y + 6, 8, self.RED, False)

            color = self.BLUE if i == self.current_option else self.WHITE
            option_surface = self.font_small.render(menu_text, True, color)
            self.screen.blit(option_surface, (menu_x, menu_y))

        # Draw wave visualization
        self.draw_wave_visualization()

        # Draw custom waveform interface if in custom mode
        self.draw_custom_waveform_interface()

        # Draw controls with icons
        controls_y = self.WINDOW_HEIGHT - 40
        controls_x = self.menu_padding

        # Draw arrow keys for navigation
        self.draw_arrow_keys_icon(controls_x, controls_y, 16, self.LIGHT_GRAY)
        nav_text = "to navigate"
        nav_surface = self.font_small.render(nav_text, True, self.LIGHT_GRAY)
        self.screen.blit(nav_surface, (controls_x + 20, controls_y))

        # Draw left/right arrows for adjustment
        adjust_x = controls_x + 120
        pygame.draw.polygon(
            self.screen,
            self.LIGHT_GRAY,
            [
                (adjust_x, controls_y + 8),
                (adjust_x + 8, controls_y + 4),
                (adjust_x + 8, controls_y + 12),
            ],
        )
        pygame.draw.polygon(
            self.screen,
            self.LIGHT_GRAY,
            [
                (adjust_x + 20, controls_y + 8),
                (adjust_x + 12, controls_y + 4),
                (adjust_x + 12, controls_y + 12),
            ],
        )
        adjust_text = "to adjust"
        adjust_surface = self.font_small.render(adjust_text, True, self.LIGHT_GRAY)
        self.screen.blit(adjust_surface, (adjust_x + 30, controls_y))

        # Draw space key for toggle
        space_x = controls_x + 200
        self.draw_space_key_icon(space_x, controls_y, 30, 16, self.LIGHT_GRAY)
        space_text = "to toggle"
        space_surface = self.font_small.render(space_text, True, self.LIGHT_GRAY)
        self.screen.blit(space_surface, (space_x + 35, controls_y))

        # Draw Q key for quit
        q_x = controls_x + 320
        self.draw_q_key_icon(q_x, controls_y, 16, self.LIGHT_GRAY)
        q_text = "to quit"
        q_surface = self.font_small.render(q_text, True, self.LIGHT_GRAY)
        self.screen.blit(q_surface, (q_x + 20, controls_y))

        pygame.display.flip()

    def draw_wave_visualization(self):
        """Draw a visual representation of the current wave"""
        wave_rect = pygame.Rect(self.WINDOW_WIDTH // 2 + 50, 150, 300, 100)
        pygame.draw.rect(self.screen, self.DARK_GRAY, wave_rect, 2)

        # Update animation
        current_time = time.time()
        self.wave_animation_offset += (current_time - self.last_frame_time) * 200
        self.last_frame_time = current_time

        # Draw wave
        points = []
        frequency = (
            self.sweep_current_frequency
            if self.sweep_active and self.sweep_current_frequency > 0
            else self.frequency
        )

        for x in range(wave_rect.width):
            # Calculate wave position
            wave_x = x / wave_rect.width
            phase = (wave_x * 4 * math.pi) + (
                self.wave_animation_offset * frequency * 0.01
            )

            if self.wave_type == "sine":
                wave_y = math.sin(phase)
            elif self.wave_type == "square":
                wave_y = 1 if math.sin(phase) > 0 else -1
            elif self.wave_type == "sawtooth":
                wave_y = 2 * ((phase / (2 * math.pi)) % 1) - 1
            elif self.wave_type == "custom":
                # Use custom waveform for visualization
                normalized_phase = phase % (2 * math.pi)
                index = int(
                    normalized_phase / (2 * math.pi) * (len(self.custom_waveform) - 1)
                )
                wave_y = self.custom_waveform[index]

            # Apply volume scaling
            wave_y *= self.volume / 100.0

            # Convert to screen coordinates
            screen_x = wave_rect.x + x
            screen_y = (
                wave_rect.y
                + wave_rect.height // 2
                - int(wave_y * wave_rect.height // 3)
            )

            points.append((screen_x, screen_y))

        # Draw the wave line
        if len(points) > 1:
            color = self.GREEN if (self.is_playing or self.sweep_active) else self.RED
            pygame.draw.lines(self.screen, color, False, points, 2)

        # Draw sweep visualization if active
        if self.sweep_active:
            self.draw_sweep_visualization()

    def draw_sweep_visualization(self):
        """Draw a visual representation of the sweep wave showing the actual frequency sweep"""
        sweep_rect = pygame.Rect(self.WINDOW_WIDTH // 2 + 50, 260, 300, 100)
        pygame.draw.rect(self.screen, self.DARK_GRAY, sweep_rect, 2)

        # Add label for sweep visualization
        sweep_label = self.font_small.render("Sweep", True, self.ORANGE)
        self.screen.blit(sweep_label, (sweep_rect.x, sweep_rect.y - 20))

        # Draw sweep wave showing the actual frequency sweep
        points = []

        for x in range(sweep_rect.width):
            # Calculate the frequency at this x position based on sweep progress
            sweep_progress = x / sweep_rect.width  # 0 to 1 across the width
            frequency = (
                self.sweep_start + (self.sweep_end - self.sweep_start) * sweep_progress
            )

            # Calculate wave position with varying frequency
            wave_x = x / sweep_rect.width

            # Use a more complex phase calculation for sweep visualization
            # This creates a chirp signal that shows the frequency change
            time_position = wave_x * 2  # Simulate time across the width
            instantaneous_frequency = (
                self.sweep_start
                + (self.sweep_end - self.sweep_start) * time_position / 2
            )

            # Calculate phase for frequency sweep (chirp signal)
            phase = (
                2
                * math.pi
                * (
                    self.sweep_start * time_position
                    + 0.5
                    * (self.sweep_end - self.sweep_start)
                    * time_position
                    * time_position
                    / 2
                )
            )

            # Add animation offset
            phase += self.wave_animation_offset * 0.01

            if self.wave_type == "sine":
                wave_y = math.sin(phase)
            elif self.wave_type == "square":
                wave_y = 1 if math.sin(phase) > 0 else -1
            elif self.wave_type == "sawtooth":
                wave_y = 2 * ((phase / (2 * math.pi)) % 1) - 1
            elif self.wave_type == "custom":
                # Use custom waveform for sweep visualization
                normalized_phase = phase % (2 * math.pi)
                index = int(
                    normalized_phase / (2 * math.pi) * (len(self.custom_waveform) - 1)
                )
                wave_y = self.custom_waveform[index]
            else:
                wave_y = math.sin(phase)  # Default fallback

            # Apply volume scaling
            wave_y *= self.volume / 100.0

            # Convert to screen coordinates
            screen_x = sweep_rect.x + x
            screen_y = (
                sweep_rect.y
                + sweep_rect.height // 2
                - int(wave_y * sweep_rect.height // 3)
            )

            points.append((screen_x, screen_y))

        # Draw the wave line
        if len(points) > 1:
            color = self.ORANGE if self.sweep_active else self.RED
            pygame.draw.lines(self.screen, color, False, points, 2)

        # Draw frequency indicators
        if self.sweep_active:
            # Draw start frequency indicator
            start_freq_text = (
                f"{self.sweep_start:.0f}Hz"
                if self.sweep_start < 1000
                else f"{self.sweep_start/1000:.1f}kHz"
            )
            start_surface = self.font_small.render(start_freq_text, True, self.ORANGE)
            self.screen.blit(
                start_surface, (sweep_rect.x, sweep_rect.y + sweep_rect.height + 5)
            )

            # Draw end frequency indicator
            end_freq_text = (
                f"{self.sweep_end:.0f}Hz"
                if self.sweep_end < 1000
                else f"{self.sweep_end/1000:.1f}kHz"
            )
            end_surface = self.font_small.render(end_freq_text, True, self.ORANGE)
            self.screen.blit(
                end_surface,
                (
                    sweep_rect.x + sweep_rect.width - end_surface.get_width(),
                    sweep_rect.y + sweep_rect.height + 5,
                ),
            )

    def handle_input(self):
        """Handle pygame input for menu navigation"""
        self.quit_requested = False
        clock = pygame.time.Clock()

        try:
            while not self.quit_requested:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit_requested = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.quit_requested = True
                        elif event.key == pygame.K_SPACE:
                            self.toggle_play_pause()
                        elif event.key == pygame.K_s:
                            self.toggle_sweep()
                        elif event.key == pygame.K_t:
                            self.toggle_play_pause()
                        elif event.key == pygame.K_UP:
                            self.current_option = (self.current_option - 1) % len(
                                self.menu_options
                            )
                        elif event.key == pygame.K_DOWN:
                            self.current_option = (self.current_option + 1) % len(
                                self.menu_options
                            )
                        elif event.key == pygame.K_LEFT:
                            self.adjust_current_option("down")
                        elif event.key == pygame.K_RIGHT:
                            self.adjust_current_option("up")
                        elif event.key == pygame.K_c:
                            if self.wave_type == "custom":
                                self.toggle_drawing_mode()
                        elif event.key == pygame.K_r:
                            if self.wave_type == "custom":
                                self.clear_custom_waveform()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.drawing_mode:
                            self.handle_mouse_drawing(event.pos, True)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if self.drawing_mode:
                            self.handle_mouse_drawing(event.pos, False)
                    elif event.type == pygame.MOUSEMOTION:
                        if self.drawing_mode and pygame.mouse.get_pressed()[0]:
                            self.handle_mouse_drawing(event.pos, True)

                # Refresh display
                self.display_menu()

                # Maintain 60 FPS for smooth animation
                clock.tick(60)

        except KeyboardInterrupt:
            pass

    def adjust_current_option(self, direction):
        """Adjust the currently selected option"""
        if self.current_option == 0:  # Play/Pause
            self.toggle_play_pause()
        elif self.current_option == 1:  # Volume
            self.adjust_volume(direction)
        elif self.current_option == 2:  # Hz Step
            self.cycle_hz_step(direction)
        elif self.current_option == 3:  # Wave Type
            self.cycle_wave_type(direction)
        elif self.current_option == 4:  # Frequency
            self.adjust_frequency(direction)
        elif self.current_option == 5:  # Sweep
            self.toggle_sweep()
        elif self.current_option == 6:  # Sweep Duration
            self.adjust_sweep_duration(direction)
        elif self.current_option == 7:  # Sweep Start
            self.adjust_sweep_start(direction)
        elif self.current_option == 8:  # Sweep End
            self.adjust_sweep_end(direction)
        elif self.current_option == 9:  # Sweep Loop
            self.toggle_sweep_loop()

    def toggle_sweep_loop(self):
        """Toggle the sweep loop on/off"""
        self.sweep_loop = not self.sweep_loop

    def toggle_drawing_mode(self):
        """Toggle drawing mode for custom waveforms"""
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.drawing_points = []

    def handle_mouse_drawing(self, pos, drawing):
        """Handle mouse drawing for custom waveforms"""
        if not drawing:
            return

        # Define drawing area (same as wave visualization area)
        drawing_rect = pygame.Rect(self.WINDOW_WIDTH // 2 + 50, 150, 300, 100)

        # Check if mouse is within drawing area
        if drawing_rect.collidepoint(pos):
            # Convert mouse position to waveform coordinates
            rel_x = pos[0] - drawing_rect.x
            rel_y = pos[1] - drawing_rect.y

            # Normalize x to 0-1 range and y to -1 to 1 range
            x_norm = rel_x / drawing_rect.width
            y_norm = (drawing_rect.height / 2 - rel_y) / (drawing_rect.height / 2)

            # Clamp y to valid range
            y_norm = max(-1.0, min(1.0, y_norm))

            # Add point to drawing
            self.drawing_points.append((x_norm, y_norm))

            # Update custom waveform when we have enough points
            if len(self.drawing_points) > 2:
                self.update_custom_waveform()

    def update_custom_waveform(self):
        """Update the custom waveform from drawing points"""
        if len(self.drawing_points) < 2:
            return

        # Sort points by x coordinate
        sorted_points = sorted(self.drawing_points, key=lambda p: p[0])

        # Interpolate to create smooth waveform
        waveform_size = 1000
        new_waveform = np.zeros(waveform_size)

        for i in range(waveform_size):
            x = i / (waveform_size - 1)

            # Find the two closest points
            left_point = None
            right_point = None

            for j in range(len(sorted_points)):
                if sorted_points[j][0] <= x:
                    left_point = sorted_points[j]
                if sorted_points[j][0] >= x and right_point is None:
                    right_point = sorted_points[j]
                    break

            # Interpolate between points
            if left_point is None:
                new_waveform[i] = sorted_points[0][1] if sorted_points else 0
            elif right_point is None:
                new_waveform[i] = sorted_points[-1][1] if sorted_points else 0
            elif left_point[0] == right_point[0]:
                new_waveform[i] = left_point[1]
            else:
                # Linear interpolation
                t = (x - left_point[0]) / (right_point[0] - left_point[0])
                new_waveform[i] = left_point[1] + t * (right_point[1] - left_point[1])

        # Make the waveform periodic by ensuring the end matches the beginning
        if len(sorted_points) > 0:
            # Smooth transition from end to beginning
            fade_length = min(50, waveform_size // 20)
            for i in range(fade_length):
                t = i / fade_length
                new_waveform[waveform_size - fade_length + i] = (
                    new_waveform[waveform_size - fade_length + i] * (1 - t)
                    + new_waveform[i] * t
                )

        self.custom_waveform = new_waveform

    def clear_custom_waveform(self):
        """Clear the custom waveform and reset to sine wave"""
        self.custom_waveform = np.sin(np.linspace(0, 2 * np.pi, 1000))
        self.drawing_points = []

    def draw_custom_waveform_interface(self):
        """Draw the custom waveform drawing interface"""
        if self.wave_type != "custom":
            return

        # Drawing area (same as wave visualization area)
        drawing_rect = pygame.Rect(self.WINDOW_WIDTH // 2 + 50, 150, 300, 100)

        # Draw different border color when in drawing mode
        border_color = self.YELLOW if self.drawing_mode else self.DARK_GRAY
        pygame.draw.rect(self.screen, border_color, drawing_rect, 3)

        # Draw instructions
        if self.drawing_mode:
            instruction_text = "Draw your waveform with the mouse!"
            instruction_color = self.YELLOW
        else:
            instruction_text = "Press 'C' to draw custom waveform"
            instruction_color = self.LIGHT_GRAY

        instruction_surface = self.font_small.render(
            instruction_text, True, instruction_color
        )
        instruction_x = (
            drawing_rect.x
            + drawing_rect.width // 2
            - instruction_surface.get_width() // 2
        )
        self.screen.blit(instruction_surface, (instruction_x, drawing_rect.y - 40))

        # Draw drawing points if in drawing mode
        if self.drawing_mode and self.drawing_points:
            for x_norm, y_norm in self.drawing_points:
                screen_x = drawing_rect.x + x_norm * drawing_rect.width
                screen_y = (
                    drawing_rect.y
                    + drawing_rect.height // 2
                    - y_norm * drawing_rect.height // 2
                )
                pygame.draw.circle(
                    self.screen, self.YELLOW, (int(screen_x), int(screen_y)), 2
                )

        # Draw additional controls
        if self.wave_type == "custom":
            controls_text = "Custom Controls: [C] Draw Mode, [R] Reset"
            controls_surface = self.font_small.render(
                controls_text, True, self.LIGHT_GRAY
            )
            controls_x = (
                drawing_rect.x
                + drawing_rect.width // 2
                - controls_surface.get_width() // 2
            )
            self.screen.blit(
                controls_surface,
                (controls_x, drawing_rect.y + drawing_rect.height + 25),
            )

    def cleanup(self):
        """Clean up resources"""
        # Stop audio stream
        if self.is_playing:
            self.stop_audio_stream()

        # Stop sweep if active
        if self.sweep_active:
            self.sweep_active = False
            if self.sweep_thread:
                self.sweep_thread.join()
        
        # Clean up pygame (GUI only)
        pygame.quit()

    def run(self):
        """Main application loop"""
        print("Initializing Wave Generator...")
        print("Make sure to run this in a terminal that supports keyboard input!")
        time.sleep(1)

        try:
            self.display_menu()
            self.handle_input()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
            print("\nThanks for using Wave Generator! 🎵")


def main():
    """Main function to run the wave generator"""
    try:
        generator = WaveGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error initializing application: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install pygame numpy keyboard")


if __name__ == "__main__":
    main()
