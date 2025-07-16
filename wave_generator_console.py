import pygame
import numpy as np
import keyboard
import threading
import time
import os
import sys

class WaveGenerator:
    def __init__(self):
        # Ultra-optimized audio settings for minimal latency and smooth playback
        pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=256)
        pygame.mixer.init()
        
        # Set up multiple channels for crossfading
        pygame.mixer.set_num_channels(8)
        
        self.sample_rate = 48000  # Match sample rate with pygame mixer for optimal performance
        self.duration = 10.0  # Duration of each wave cycle in seconds
        self.is_playing = False
        self.current_sound = None
        self.sound_thread = None
        self.stop_sound = False
        
        # For smooth transitions
        self.current_phase = 0.0
        self.phase_lock = threading.Lock()
        self.current_channel = 0
        self.transition_in_progress = False
        self.playback_start_time = None
        
        # Menu options
        self.menu_options = ["Tone Playback", "Volume (%)", "Hz Step", "Wave Type", "Frequency (Hz)", "Sweep", "Sweep Duration", "Sweep Start", "Sweep End", "Sweep Loop"]
        self.current_option = 0
        
        # Settings
        self.frequency = 440.0  # Default A4 note
        self.volume = 50      # 50% volume
        self.wave_type = "sine"  # Default wave type
        self.wave_types = ["sine", "square", "sawtooth"]
        
        # Hz step options
        self.hz_steps = [0.1, 1.0, 10.0, 100.0, 1000.0]
        self.current_hz_step_index = 4  # Start with 1kHz steps
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
        self.sweep_durations = [1.0, 2.0, 5.0, 10.0, 20.0]  # Available durations
        self.current_sweep_duration_index = 0  # default to 1 second
        self.sweep_current_frequency = 0.0  # Current frequency during sweep
        self.user_selected_frequency = 440.0  # User's selected frequency (preserved during sweep)
        self.sweep_loop = True  # Whether sweep should loop continuously (default on)
        
    def generate_wave(self, wave_type, frequency, volume, duration):
        """Generate a wave of specified type, frequency, volume, and duration"""
        frames = int(duration * self.sample_rate)
        arr = np.zeros((frames, 2))
        
        for i in range(frames):
            t = float(i) / self.sample_rate
            
            if wave_type == "sine":
                wave_value = np.sin(2 * np.pi * frequency * t)
            elif wave_type == "square":
                wave_value = np.sign(np.sin(2 * np.pi * frequency * t))
            elif wave_type == "sawtooth":
                wave_value = 2 * (t * frequency - np.floor(t * frequency + 0.5))
            
            # Apply volume (0-100 scale)
            wave_value *= (volume / 100.0)
            
            # Convert to 16-bit integer range
            wave_value = int(wave_value * 32767)
            
            # Stereo output
            arr[i][0] = wave_value
            arr[i][1] = wave_value
        
        return arr.astype(np.int16)
    
    def generate_wave_chunk(self, wave_type, frequency, volume, frames, start_phase):
        """Generate a wave chunk with specified starting phase for seamless continuity"""
        arr = np.zeros((frames, 2), dtype=np.int16)
        
        # Use numpy arrays for better performance and precision
        i_array = np.arange(frames, dtype=np.float64)
        phase_array = start_phase + (i_array / self.sample_rate) * frequency * 2 * np.pi
        
        if wave_type == "sine":
            wave_values = np.sin(phase_array)
        elif wave_type == "square":
            wave_values = np.sign(np.sin(phase_array))
        elif wave_type == "sawtooth":
            # Normalize phase to [0, 2π] for sawtooth calculation
            normalized_phase = phase_array % (2 * np.pi)
            wave_values = 2 * (normalized_phase / (2 * np.pi)) - 1
        else:
            # Default to sine wave if unknown type
            wave_values = np.sin(phase_array)
        
        # Apply volume (0-100 scale)
        wave_values *= (volume / 100.0)
        
        # Apply gentle anti-aliasing filter to reduce high frequency artifacts
        # Simple low-pass filter using a moving average
        if len(wave_values) > 3:
            wave_values_filtered = wave_values.copy()
            wave_values_filtered[1:-1] = (wave_values[:-2] + 2 * wave_values[1:-1] + wave_values[2:]) / 4
            wave_values = wave_values_filtered
        
        # Convert to 16-bit integer range with proper clipping
        wave_values_int = np.clip(wave_values * 32767, -32767, 32767).astype(np.int16)
        
        # Stereo output
        arr[:, 0] = wave_values_int
        arr[:, 1] = wave_values_int
        
        return arr
    
    def create_crossfade_transition(self, old_wave_type, old_frequency, old_volume, new_wave_type, new_frequency, new_volume, duration=0.15):
        """Create a smooth crossfade transition between two wave configurations"""
        frames = int(duration * self.sample_rate)
        arr = np.zeros((frames, 2))
        
        # Get current phase for continuity
        with self.phase_lock:
            start_phase = self.current_phase
        
        for i in range(frames):
            # Calculate progress through transition (0 to 1)
            progress = i / frames
            
            # Calculate phases for both waves
            old_phase = start_phase + (i / self.sample_rate) * old_frequency * 2 * np.pi
            new_phase = start_phase + (i / self.sample_rate) * new_frequency * 2 * np.pi
            
            # Generate old wave value
            if old_wave_type == "sine":
                old_value = np.sin(old_phase)
            elif old_wave_type == "square":
                old_value = np.sign(np.sin(old_phase))
            elif old_wave_type == "sawtooth":
                normalized_phase = old_phase % (2 * np.pi)
                old_value = 2 * (normalized_phase / (2 * np.pi)) - 1
            
            # Generate new wave value
            if new_wave_type == "sine":
                new_value = np.sin(new_phase)
            elif new_wave_type == "square":
                new_value = np.sign(np.sin(new_phase))
            elif new_wave_type == "sawtooth":
                normalized_phase = new_phase % (2 * np.pi)
                new_value = 2 * (normalized_phase / (2 * np.pi)) - 1
            
            # Smooth crossfade using cosine interpolation
            fade_factor = 0.5 * (1 - np.cos(progress * np.pi))
            
            # Apply volume crossfade as well
            old_volume_factor = (old_volume / 100.0) * (1 - fade_factor)
            new_volume_factor = (new_volume / 100.0) * fade_factor
            
            # Mix the two signals with their respective volumes
            wave_value = old_value * old_volume_factor + new_value * new_volume_factor
            
            # Convert to 16-bit integer range
            wave_value = int(wave_value * 32767)
            
            # Stereo output
            arr[i][0] = wave_value
            arr[i][1] = wave_value
        
        # Update phase for next generation
        with self.phase_lock:
            self.current_phase = (start_phase + (frames / self.sample_rate) * new_frequency * 2 * np.pi) % (2 * np.pi)
        
        return arr.astype(np.int16)
    
    def sweep_frequency(self, sweep_duration=None, sweep_start=None, sweep_end=None):
        """Sweep frequency over a specified range and duration"""
        sweep_duration = sweep_duration or self.sweep_duration
        sweep_start = sweep_start or self.sweep_start
        sweep_end = sweep_end or self.sweep_end

        self.user_selected_frequency = self.frequency
        steps = int(sweep_duration * 50)
        step_duration = sweep_duration / steps
        frequency_step = (sweep_end - sweep_start) / steps

        while True:
            for i in range(steps):
                if not self.sweep_active:
                    break
                self.sweep_current_frequency = sweep_start + i * frequency_step
                # Don't modify self.frequency - sweep should be independent
                time.sleep(step_duration)
            if not self.sweep_loop or not self.sweep_active:
                break

        self.sweep_active = False
        self.sweep_current_frequency = 0.0

    def play_sweep_sound(self):
        """Play sweep sound in separate channel with real-time frequency updates"""
        channel_b = pygame.mixer.Channel(1)
        channel_b.set_volume(1.0)
        
        # Generate initial short sound chunk
        chunk_duration = 0.1  # 100ms chunks for responsive frequency changes
        chunk_frames = int(chunk_duration * self.sample_rate)
        current_phase = 0.0
        
        while self.sweep_active:
            try:
                # Use current sweep frequency if available, otherwise use sweep start
                current_freq = self.sweep_current_frequency if self.sweep_current_frequency > 0 else self.sweep_start
                
                # Generate wave chunk with current frequency
                wave_data = self.generate_wave_chunk(self.wave_type, current_freq, self.volume, chunk_frames, current_phase)
                sweep_sound = pygame.sndarray.make_sound(wave_data)
                
                # Update phase for continuity
                current_phase = (current_phase + (chunk_frames / self.sample_rate) * current_freq * 2 * np.pi) % (2 * np.pi)
                
                # Play the chunk
                channel_b.play(sweep_sound)
                
                # Wait for chunk to finish
                time.sleep(chunk_duration)
                
            except Exception as e:
                print(f"Error in sweep sound generation: {e}")
                break
        
        channel_b.stop()
    
    def toggle_sweep(self):
        """Toggle frequency sweep on/off"""
        if self.sweep_active:
            self.sweep_active = False
            if self.sweep_thread:
                self.sweep_thread.join()
        else:
            self.sweep_active = True
            # Start both sweep frequency control and sound generation
            self.sweep_thread = threading.Thread(target=self.sweep_frequency)
            self.sweep_thread.daemon = True
            self.sweep_thread.start()
            
            # Start sweep sound generation in separate thread
            self.sweep_sound_thread = threading.Thread(target=self.play_sweep_sound)
            self.sweep_sound_thread.daemon = True
            self.sweep_sound_thread.start()
    
    def adjust_sweep_duration(self, direction):
        """Adjust sweep duration up or down"""
        if direction == "up":
            self.current_sweep_duration_index = (self.current_sweep_duration_index + 1) % len(self.sweep_durations)
        elif direction == "down":
            self.current_sweep_duration_index = (self.current_sweep_duration_index - 1) % len(self.sweep_durations)
        self.sweep_duration = self.sweep_durations[self.current_sweep_duration_index]
        
        # If sweep is currently active, restart it with the new duration
        if self.sweep_active:
            self.sweep_active = False
            if self.sweep_thread:
                self.sweep_thread.join()
            # Restart sweep with new duration
            self.sweep_active = True
            self.sweep_thread = threading.Thread(target=self.sweep_frequency)
            self.sweep_thread.daemon = True
            self.sweep_thread.start()
    
    def adjust_sweep_start(self, direction):
        """Adjust sweep start frequency up or down"""
        if direction == "up":
            self.sweep_start = min(self.sweep_start + self.current_hz_step, self.sweep_end - 1.0)
        elif direction == "down":
            self.sweep_start = max(self.sweep_start - self.current_hz_step, self.min_frequency)
        
        # Round to appropriate decimal places
        if self.current_hz_step >= 1.0:
            self.sweep_start = round(self.sweep_start, 1)
        else:
            self.sweep_start = round(self.sweep_start, 1)
        
        # If sweep is currently active, restart it with the new start frequency
        if self.sweep_active:
            self.sweep_active = False
            if self.sweep_thread:
                self.sweep_thread.join()
            # Restart sweep with new start frequency
            self.sweep_active = True
            self.sweep_thread = threading.Thread(target=self.sweep_frequency)
            self.sweep_thread.daemon = True
            self.sweep_thread.start()
    
    def adjust_sweep_end(self, direction):
        """Adjust sweep end frequency up or down"""
        if direction == "up":
            self.sweep_end = min(self.sweep_end + self.current_hz_step, self.max_frequency)
        elif direction == "down":
            self.sweep_end = max(self.sweep_end - self.current_hz_step, self.sweep_start + 1.0)
        
        # Round to appropriate decimal places
        if self.current_hz_step >= 1.0:
            self.sweep_end = round(self.sweep_end, 1)
        else:
            self.sweep_end = round(self.sweep_end, 1)
        
        # If sweep is currently active, restart it with the new end frequency
        if self.sweep_active:
            self.sweep_active = False
            if self.sweep_thread:
                self.sweep_thread.join()
            # Restart sweep with new end frequency
            self.sweep_active = True
            self.sweep_thread = threading.Thread(target=self.sweep_frequency)
            self.sweep_thread.daemon = True
            self.sweep_thread.start()

    def play_continuous_sound(self):
        """Play continuous tone sound"""
        try:
            loop_duration = 2.0  # 2 seconds duration
            loop_frames = int(loop_duration * self.sample_rate)
            
            current_params = (self.frequency, self.volume, self.wave_type)
            
            with self.phase_lock:
                self.current_phase = 0.0
                self.playback_start_time = time.time()
            
            wave_data = self.generate_wave_chunk(self.wave_type, self.frequency, self.volume, loop_frames, 0.0)
            current_sound = pygame.sndarray.make_sound(wave_data)
        except Exception as e:
            print(f"Error in sound generation: {e}")
            return
        
        channel_a = pygame.mixer.Channel(0)
        active_channel = channel_a
        active_channel.set_volume(1.0)
        
        time.sleep(0.05)

        active_channel.play(current_sound, loops=-1)
        
        while not self.stop_sound:
            new_params = (self.frequency, self.volume, self.wave_type)
            if new_params != current_params and not self.transition_in_progress:
                self.transition_in_progress = True
                
                old_frequency, old_volume, old_wave_type = current_params
                new_frequency, new_volume, new_wave_type = new_params
                
                current_time = time.time()
                with self.phase_lock:
                    elapsed_time = current_time - self.playback_start_time
                    transition_start_phase = (elapsed_time * old_frequency * 2 * np.pi) % (2 * np.pi)
                    self.current_phase = transition_start_phase
                    self.playback_start_time = current_time
                
                
                # Generate new sound with updated parameters and phase continuity
                new_wave_data = self.generate_wave_chunk(new_wave_type, new_frequency,
                                                        new_volume, loop_frames, transition_start_phase)
                new_sound = pygame.sndarray.make_sound(new_wave_data)
                
                # Get the inactive channel for the new sound
                channel_b = pygame.mixer.Channel(1)
                inactive_channel = channel_b if active_channel == channel_a else channel_a
                
                # Start new sound on inactive channel at low volume
                inactive_channel.set_volume(0.0)
                inactive_channel.play(new_sound, loops=-1)
                
                # Wait a tiny bit to ensure new sound is properly buffered
                time.sleep(0.002)
                
                # Perform ultra-smooth crossfade with precise timing
                crossfade_steps = 40  # More steps for ultra-smooth transition
                step_duration = 0.0005  # 0.5ms per step = 20ms total crossfade
                
                for step in range(crossfade_steps + 1):
                    if self.stop_sound:
                        break
                        
                    progress = step / crossfade_steps
                    
                    # Use cubic hermite interpolation for extremely smooth transitions
                    # This provides better continuity than smoothstep
                    fade_factor = progress * progress * (3.0 - 2.0 * progress)
                    # Apply additional smoothing with a slight S-curve enhancement
                    fade_factor = fade_factor * fade_factor * (3.0 - 2.0 * fade_factor)
                    
                    old_volume = 1.0 - fade_factor
                    new_volume_level = fade_factor
                    
                    # Apply volumes with slight easing to prevent abrupt changes
                    active_channel.set_volume(max(0.0, min(1.0, old_volume)))
                    inactive_channel.set_volume(max(0.0, min(1.0, new_volume_level)))
                    
                    time.sleep(step_duration)
                
                # Complete the switch
                if not self.stop_sound:
                    active_channel.stop()  # Stop the old sound
                    inactive_channel.set_volume(1.0)  # Ensure new sound is at full volume
                    
                    # Swap active channel
                    active_channel = inactive_channel
                    current_sound = new_sound
                    current_params = new_params
                    
                    # Update phase tracking for the new loop
                    with self.phase_lock:
                        self.current_phase = (transition_start_phase + (loop_frames / self.sample_rate) * new_frequency * 2 * np.pi) % (2 * np.pi)
                
                self.transition_in_progress = False
            
            time.sleep(0.01)  # Very fast checking for ultra-responsive transitions
        
        # Stop all playback
        channel_a.stop()
        channel_b = pygame.mixer.Channel(1)
        channel_b.stop()
    
    def toggle_play_pause(self):
        """Toggle play/pause state"""
        if self.is_playing:
            self.stop_sound = True
            if self.sound_thread:
                self.sound_thread.join()
            pygame.mixer.stop()
            self.is_playing = False
        else:
            self.stop_sound = False
            self.is_playing = True
            self.sound_thread = threading.Thread(target=self.play_continuous_sound)
            self.sound_thread.daemon = True
            self.sound_thread.start()
    
    def adjust_frequency(self, direction):
        """Adjust frequency up or down using current Hz step"""
        # If sweep is active, adjust the user's selected frequency (preserved)
        # Otherwise, adjust the current frequency
        if self.sweep_active:
            # Adjust user's selected frequency (what they'll get when sweep ends)
            if direction == "up":
                self.user_selected_frequency = min(self.user_selected_frequency + self.current_hz_step, self.max_frequency)
            elif direction == "down":
                self.user_selected_frequency = max(self.user_selected_frequency - self.current_hz_step, self.min_frequency)
            
            # Round to appropriate decimal places based on step size
            if self.current_hz_step >= 1.0:
                self.user_selected_frequency = round(self.user_selected_frequency, 1)
            else:
                self.user_selected_frequency = round(self.user_selected_frequency, 1)
        else:
            # Normal frequency adjustment when sweep is not active
            if direction == "up":
                self.frequency = min(self.frequency + self.current_hz_step, self.max_frequency)
            elif direction == "down":
                self.frequency = max(self.frequency - self.current_hz_step, self.min_frequency)
            
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
            self.volume = max(self.volume - 5, 0)    # Min 0%
    
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
            self.current_hz_step_index = (self.current_hz_step_index + 1) % len(self.hz_steps)
        elif direction == "down":
            self.current_hz_step_index = (self.current_hz_step_index - 1) % len(self.hz_steps)
        self.current_hz_step = self.hz_steps[self.current_hz_step_index]
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_menu(self):
        """Display the current menu state"""
        self.clear_screen()
        print("🎵 Wave Generator 🎵")
        print("=" * 30)
        print()
        
        # Show overall status
        print("Status:")
        is_anything_playing = self.is_playing or self.sweep_active
        status_icon = "🔊" if is_anything_playing else "⏸️"
        status_text = "Playing" if is_anything_playing else "Paused"
        print(f"  {status_icon} {status_text}")
        print()
        
        for i, option in enumerate(self.menu_options):
            prefix = "► " if i == self.current_option else "  "
            
            if i == 0:  # Play/Pause
                status = "🔊 Playing" if self.is_playing else "⏸️  Paused"
                print(f"{prefix}{option}: {status}")
            elif i == 1:  # Volume
                volume_bar = "█" * (self.volume // 5) + "░" * (20 - self.volume // 5)
                print(f"{prefix}{option}: {self.volume}% [{volume_bar}]")
            elif i == 2:  # Hz Step
                # Show all Hz steps with current selection in brackets
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
                
                print(f"{prefix}{option}: {', '.join(hz_step_displays)}")
            elif i == 3:  # Wave Type
                # Show all wave types with current selection in brackets
                wave_type_displays = []
                for wave_type in self.wave_types:
                    if wave_type == self.wave_type:
                        wave_type_displays.append(f"[{wave_type.capitalize()}]")
                    else:
                        wave_type_displays.append(wave_type.capitalize())
                
                print(f"{prefix}{option}: {', '.join(wave_type_displays)}")
            elif i == 4:  # Frequency
                if self.frequency >= 1000:
                    freq_display = f"{self.frequency/1000:.1f}kHz"
                else:
                    freq_display = f"{self.frequency:.1f}Hz"
                print(f"{prefix}{option}: {freq_display}")
            elif i == 5:  # Sweep
                sweep_status = "🔄 Active" if self.sweep_active else "⏸️  Inactive"
                if self.sweep_active and self.sweep_current_frequency > 0:
                    if self.sweep_current_frequency >= 1000:
                        current_display = f"{self.sweep_current_frequency/1000:.1f}kHz"
                    else:
                        current_display = f"{self.sweep_current_frequency:.1f}Hz"
                    print(f"{prefix}{option}: {sweep_status} (Current: {current_display})")
                else:
                    print(f"{prefix}{option}: {sweep_status}")
            elif i == 6:  # Sweep Duration
                # Show all sweep durations with current selection in brackets
                duration_displays = []
                for j, duration in enumerate(self.sweep_durations):
                    if j == self.current_sweep_duration_index:
                        duration_displays.append(f"[{duration}s]")
                    else:
                        duration_displays.append(f"{duration}s")
                
                print(f"{prefix}{option}: {', '.join(duration_displays)}")
            elif i == 7:  # Sweep Start
                if self.sweep_start >= 1000:
                    start_display = f"{self.sweep_start/1000:.1f}kHz"
                else:
                    start_display = f"{self.sweep_start:.1f}Hz"
                print(f"{prefix}{option}: {start_display}")
            elif i == 8:  # Sweep End
                if self.sweep_end >= 1000:
                    end_display = f"{self.sweep_end/1000:.1f}kHz"
                else:
                    end_display = f"{self.sweep_end:.1f}Hz"
                print(f"{prefix}{option}: {end_display}")
            elif i == 9:  # Sweep Loop
                loop_status = "🔄 On" if self.sweep_loop else "⏸️  Off"
                print(f"{prefix}{option}: {loop_status}")
        
        print()
        print("Controls:")
        print("↑/↓ Arrow keys: Navigate menu")
        print("←/→ Arrow keys: Adjust selected option")
        print("Space: Toggle Tone Play/Pause")
        print("s: Toggle Sweep Play/Pause | t: Toggle Tone Play/Pause")
        print("Q: Quit")
    
    def handle_input(self):
        """Handle keyboard input with ultra-fast response"""
        menu_needs_refresh = True
        last_key_time = {}
        debounce_delay = 0.05  # Ultra-fast debounce for instant response
        
        # Set up event-driven keyboard handlers for instant response
        def on_key_event(event):
            nonlocal menu_needs_refresh, last_key_time
            
            if event.event_type != keyboard.KEY_DOWN:
                return
            
            current_time = time.time()
            key = event.name
            
            # Arrow keys are now working with Finnish keyboard layout
            
            # Ultra-fast debouncing - only check for repeated presses
            if key in last_key_time and (current_time - last_key_time[key]) < debounce_delay:
                return
            
            last_key_time[key] = current_time
            
            # Handle keys immediately without polling delays
            if key == 'q':
                self.quit_requested = True
            elif key == 'space':
                self.toggle_play_pause()
                menu_needs_refresh = True
            elif key == 's':
                self.toggle_sweep()
                menu_needs_refresh = True
            elif key == 't':
                self.toggle_play_pause()
                menu_needs_refresh = True
            elif key in ['up', 'up arrow', 'ylänuoli']:
                self.current_option = (self.current_option - 1) % len(self.menu_options)
                menu_needs_refresh = True
            elif key in ['down', 'down arrow', 'alanuoli']:
                self.current_option = (self.current_option + 1) % len(self.menu_options)
                menu_needs_refresh = True
            elif key in ['left', 'left arrow', 'vasen nuoli']:
                self.adjust_current_option("down")
                menu_needs_refresh = True
            elif key in ['right', 'right arrow', 'oikea nuoli']:
                self.adjust_current_option("up")
                menu_needs_refresh = True
        
        # Register event-driven keyboard handler
        keyboard.hook(on_key_event)
        
        self.quit_requested = False
        
        try:
            while not self.quit_requested:
                # Only refresh menu when needed
                self.display_menu()
                menu_needs_refresh = False

                # Minimal sleep to prevent excessive CPU usage while maintaining responsiveness
                time.sleep(0.017)  # ~10 Hz update rate for display
                
        except KeyboardInterrupt:
            pass
        finally:
            keyboard.unhook_all()
    
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
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_playing:
            self.stop_sound = True
            if self.sound_thread:
                self.sound_thread.join()
        
        # Stop sweep if active
        if self.sweep_active:
            self.sweep_active = False
            if self.sweep_thread:
                self.sweep_thread.join()
        
        pygame.mixer.quit()
    
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
