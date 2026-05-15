import numpy as np
import threading
import time


class WaveEngine:
    """Shared wave generation engine for both console and GUI versions"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        
        # For smooth transitions
        self.current_phase = 0.0
        self.phase_lock = threading.Lock()
        self.transition_in_progress = False
        self.playback_start_time = None
        
        # Wave types supported by the engine
        self.wave_types = ["sine", "square", "sawtooth", "custom"]
        
        # Custom waveform data (default to sine wave)
        self.custom_waveform = np.sin(np.linspace(0, 2 * np.pi, 1000))
        
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
            elif wave_type == "custom":
                # Use custom waveform by indexing into the custom_waveform array
                phase = (2 * np.pi * frequency * t) % (2 * np.pi)
                index = int(phase / (2 * np.pi) * (len(self.custom_waveform) - 1))
                wave_value = self.custom_waveform[index]
            else:
                # Default to sine wave if unknown type
                wave_value = np.sin(2 * np.pi * frequency * t)

            # Apply volume (0-100 scale)
            wave_value *= volume / 100.0

            # Convert to 16-bit integer range
            wave_value = int(wave_value * 32767)

            # Stereo output
            arr[i][0] = wave_value
            arr[i][1] = wave_value

        return arr.astype(np.int16)

    def generate_wave_chunk(self, wave_type, frequency, volume, frames, start_phase):
        """Generate a wave chunk with specified starting phase for seamless continuity"""
        arr = np.zeros((frames, 2), dtype=np.int16)
        
        # Generate samples ensuring perfect phase continuity
        phase_increment = (frequency * 2 * np.pi) / self.sample_rate
        
        # Use sample-by-sample generation for maximum precision
        for i in range(frames):
            phase = start_phase + i * phase_increment
            
            if wave_type == "sine":
                wave_value = np.sin(phase)
            elif wave_type == "square":
                wave_value = np.sign(np.sin(phase))
            elif wave_type == "sawtooth":
                # Use fmod for precise sawtooth generation
                cycle_position = (phase / (2 * np.pi)) % 1.0
                wave_value = 2.0 * cycle_position - 1.0
            elif wave_type == "custom":
                normalized_phase = phase % (2 * np.pi)
                index = int(normalized_phase / (2 * np.pi) * (len(self.custom_waveform) - 1))
                # Ensure index is within bounds
                index = max(0, min(index, len(self.custom_waveform) - 1))
                wave_value = self.custom_waveform[index]
            else:
                wave_value = np.sin(phase)
            
            # Apply volume and convert to 16-bit
            wave_value *= volume / 100.0
            wave_value_int = int(np.clip(wave_value * 32767, -32767, 32767))
            
            arr[i, 0] = wave_value_int
            arr[i, 1] = wave_value_int

        return arr

    def create_crossfade_transition(
        self,
        old_wave_type,
        old_frequency,
        old_volume,
        new_wave_type,
        new_frequency,
        new_volume,
        duration=0.15,
    ):
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
            old_value = self._get_wave_value(old_wave_type, old_phase)
            
            # Generate new wave value
            new_value = self._get_wave_value(new_wave_type, new_phase)

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
            self.current_phase = (
                start_phase + (frames / self.sample_rate) * new_frequency * 2 * np.pi
            ) % (2 * np.pi)

        return arr.astype(np.int16)

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
        """Generate seamless waveform transition using perfect phase continuity"""
        # Generate the new waveform with perfect phase continuity from the current phase
        return self.generate_wave_chunk(new_wave_type, new_frequency, new_volume, loop_frames, start_phase)
    
    def _generate_wave_values(self, wave_type, phase_array):
        """Generate waveform values for given phase array using vectorized operations"""
        if wave_type == "sine":
            return np.sin(phase_array)
        elif wave_type == "square":
            return np.sign(np.sin(phase_array))
        elif wave_type == "sawtooth":
            normalized_phase = phase_array % (2 * np.pi)
            return 2 * (normalized_phase / (2 * np.pi)) - 1
        elif wave_type == "custom":
            normalized_phase = phase_array % (2 * np.pi)
            indices = (normalized_phase / (2 * np.pi) * (len(self.custom_waveform) - 1)).astype(int)
            # Ensure indices are within bounds
            indices = np.clip(indices, 0, len(self.custom_waveform) - 1)
            return self.custom_waveform[indices]
        else:
            return np.sin(phase_array)

    def _get_wave_value(self, wave_type, phase):
        """Get wave value for a given wave type and phase"""
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
            return self.custom_waveform[index]
        else:
            return np.sin(phase)
    
    def _calculate_phase_offset(self, wave_type, frequency, target_amplitude, volume):
        """Calculate phase offset needed to match target amplitude for seamless transition"""
        if volume == 0:
            return 0.0  # Avoid division by zero
        
        # Normalize target amplitude by volume
        normalized_target = target_amplitude / (volume / 100.0)
        
        # Clamp to valid range [-1, 1]
        normalized_target = np.clip(normalized_target, -1.0, 1.0)
        
        if wave_type == "sine":
            # For sine wave, use arcsin to find the phase that gives the target amplitude
            try:
                # Handle edge cases where normalized_target might be slightly out of [-1,1] due to floating point precision
                phase_offset = np.arcsin(normalized_target)
                # Choose the phase that results in positive frequency derivative (upward slope)
                # This provides smoother transitions in most cases
                return phase_offset
            except:
                return 0.0
        elif wave_type == "square":
            # For square wave, match the sign
            if normalized_target >= 0:
                return 0.0  # Positive part of square wave
            else:
                return np.pi  # Negative part of square wave
        elif wave_type == "sawtooth":
            # For sawtooth wave, find phase that gives target amplitude
            # sawtooth = 2 * (phase / (2π)) - 1
            # Solve for phase: phase = π * (normalized_target + 1)
            phase_offset = np.pi * (normalized_target + 1)
            return phase_offset % (2 * np.pi)
        elif wave_type == "custom":
            # For custom waveform, find the closest matching amplitude
            differences = np.abs(self.custom_waveform - normalized_target)
            min_index = np.argmin(differences)
            phase_offset = (min_index / len(self.custom_waveform)) * 2 * np.pi
            return phase_offset
        else:
            # Default to sine wave calculation
            try:
                return np.arcsin(normalized_target)
            except:
                return 0.0

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

    def set_custom_waveform(self, waveform):
        """Set a custom waveform for the 'custom' wave type"""
        self.custom_waveform = waveform

    def get_custom_waveform(self):
        """Get the current custom waveform"""
        return self.custom_waveform.copy()

    def reset_custom_waveform(self):
        """Reset custom waveform to default sine wave"""
        self.custom_waveform = np.sin(np.linspace(0, 2 * np.pi, 1000))
