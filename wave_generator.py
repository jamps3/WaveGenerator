from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
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


class WaveGenerator(App):
    def __init__(self):
        super().__init__()
        # Professional audio settings
        self.audio_sample_rate = 44100
        self.audio_block_size = 512

        # Audio state for sounddevice
        self.audio_stream = None
        self.audio_phase = 0.0
        self.audio_lock = threading.Lock()
        self.audio_running = False

        # Window settings
        Window.size = (1200, 800)
        Window.title = "Wave Generator"

        # Colors (Kivy uses 0-1 range)
        self.BLACK = (0, 0, 0, 1)
        self.WHITE = (1, 1, 1, 1)
        self.BLUE = (0.39, 0.58, 0.93, 1)
        self.GREEN = (0.24, 0.7, 0.44, 1)
        self.RED = (0.86, 0.08, 0.24, 1)
        self.YELLOW = (1, 0.84, 0, 1)
        self.GRAY = (0.5, 0.5, 0.5, 1)
        self.LIGHT_GRAY = (0.83, 0.83, 0.83, 1)
        self.DARK_GRAY = (0.25, 0.25, 0.25, 1)
        self.ORANGE = (1, 0.65, 0, 1)

        # Animation variables
        self.wave_animation_offset = 0
        self.last_frame_time = time.time()

        # UI elements
        self.edit_mode = 'points'  # 'points' or 'free'

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
        self.edit_mode = 'points'  # 'points' or 'free'
        self.control_points = [(i / 9.0, math.sin(i / 9.0 * 2 * math.pi)) for i in range(10)]
        self.dragging_point = None
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

    def build(self):
        # Root layout
        root = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))

        # Title bar
        title_bar = BoxLayout(size_hint_y=0.08, padding=dp(5))
        title = Label(text="Wave Generator", font_size=dp(30), color=self.YELLOW, bold=True, halign='center')
        title_bar.add_widget(title)
        root.add_widget(title_bar)

        # Status bar
        self.status_label = Label(text="Status: Paused", font_size=dp(20), color=self.RED, size_hint_y=0.05, halign='center')
        root.add_widget(self.status_label)

        # Main content area
        main = BoxLayout(orientation='horizontal', size_hint_y=0.87, spacing=dp(10))

        # Left panel: Controls
        controls_scroll = ScrollView(size_hint_x=0.4)
        controls = GridLayout(cols=1, size_hint_y=None, spacing=dp(5), padding=dp(5))
        controls.bind(minimum_height=controls.setter('height'))

        # Play/Pause
        play_btn = Button(text="Play/Pause", font_size=dp(16), background_color=self.GREEN, size_hint_y=None, height=dp(40))
        play_btn.bind(on_press=self.on_play_pause)
        controls.add_widget(Label(text="Playback:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(play_btn)

        # Volume
        vol_label = Label(text=f"Volume: {self.volume}%", font_size=dp(14), color=self.WHITE)
        self.volume_slider = Slider(min=0, max=100, value=self.volume, size_hint_y=None, height=dp(40))
        self.volume_slider.bind(on_touch_up=self.on_volume_change)
        controls.add_widget(vol_label)
        controls.add_widget(self.volume_slider)

        # Frequency
        freq_label = Label(text=f"Frequency: {self.frequency:.1f} Hz", font_size=dp(14), color=self.WHITE)
        freq_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        freq_down = Button(text="-", font_size=dp(16))
        freq_down.bind(on_press=self.on_freq_down)
        freq_up = Button(text="+", font_size=dp(16))
        freq_up.bind(on_press=self.on_freq_up)
        freq_box.add_widget(freq_down)
        freq_box.add_widget(freq_label)
        freq_box.add_widget(freq_up)
        controls.add_widget(Label(text="Frequency:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(freq_box)

        # Hz Step
        hz_step_label = Label(text=f"Hz Step: {self.current_hz_step}", font_size=dp(14), color=self.WHITE)
        hz_step_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        hz_down = Button(text="<", font_size=dp(16))
        hz_down.bind(on_press=self.on_hz_down)
        hz_up = Button(text=">", font_size=dp(16))
        hz_up.bind(on_press=self.on_hz_up)
        hz_down.bind(on_press=self.on_hz_down)
        hz_up.bind(on_press=self.on_hz_up)
        hz_step_box.add_widget(hz_down)
        hz_step_box.add_widget(hz_step_label)
        hz_step_box.add_widget(hz_up)
        controls.add_widget(Label(text="Hz Step:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(hz_step_box)

        # Wave Type
        wave_type_label = Label(text=f"Wave Type: {self.wave_type.capitalize()}", font_size=dp(14), color=self.WHITE)
        wave_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        wave_down = Button(text="<", font_size=dp(16))
        wave_down.bind(on_press=self.on_wave_down)
        wave_up = Button(text=">", font_size=dp(16))
        wave_up.bind(on_press=self.on_wave_up)
        wave_box.add_widget(wave_down)
        wave_box.add_widget(wave_type_label)
        wave_box.add_widget(wave_up)
        controls.add_widget(Label(text="Wave Type:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(wave_box)

        # Sweep
        sweep_btn = Button(text="Toggle Sweep", font_size=dp(16), background_color=self.BLUE, size_hint_y=None, height=dp(40))
        sweep_btn.bind(on_press=self.on_sweep_toggle)
        controls.add_widget(Label(text="Sweep:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(sweep_btn)

        # Sweep Duration
        sweep_dur_label = Label(text=f"Sweep Duration: {self.sweep_durations[self.current_sweep_duration_index]}s", font_size=dp(14), color=self.WHITE)
        sweep_dur_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        dur_down = Button(text="<", font_size=dp(16))
        dur_down.bind(on_press=self.on_sweep_dur_down)
        dur_up = Button(text=">", font_size=dp(16))
        dur_up.bind(on_press=self.on_sweep_dur_up)
        sweep_dur_box.add_widget(dur_down)
        sweep_dur_box.add_widget(sweep_dur_label)
        sweep_dur_box.add_widget(dur_up)
        controls.add_widget(Label(text="Sweep Duration:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(sweep_dur_box)

        # Sweep Start
        sweep_start_label = Label(text=f"Sweep Start: {self.sweep_start:.1f} Hz", font_size=dp(14), color=self.WHITE)
        sweep_start_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        start_down = Button(text="<", font_size=dp(16))
        start_down.bind(on_press=self.on_sweep_start_down)
        start_up = Button(text=">", font_size=dp(16))
        start_up.bind(on_press=self.on_sweep_start_up)
        sweep_start_box.add_widget(start_down)
        sweep_start_box.add_widget(sweep_start_label)
        sweep_start_box.add_widget(start_up)
        controls.add_widget(Label(text="Sweep Start:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(sweep_start_box)

        # Sweep End
        sweep_end_label = Label(text=f"Sweep End: {self.sweep_end:.1f} Hz", font_size=dp(14), color=self.WHITE)
        sweep_end_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        end_down = Button(text="<", font_size=dp(16))
        end_down.bind(on_press=self.on_sweep_end_down)
        end_up = Button(text=">", font_size=dp(16))
        end_up.bind(on_press=self.on_sweep_end_up)
        sweep_end_box.add_widget(end_down)
        sweep_end_box.add_widget(sweep_end_label)
        sweep_end_box.add_widget(end_up)
        controls.add_widget(Label(text="Sweep End:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(sweep_end_box)

        # Sweep Loop
        self.loop_btn = Button(text="Loop Off", font_size=dp(16), background_color=self.ORANGE, size_hint_y=None, height=dp(40))
        self.loop_btn.bind(on_press=self.on_loop_toggle)
        controls.add_widget(Label(text="Sweep Loop:", font_size=dp(14), color=self.WHITE))
        controls.add_widget(self.loop_btn)

        controls_scroll.add_widget(controls)
        main.add_widget(controls_scroll)

        # Right panel: Visualization
        viz_area = BoxLayout(orientation='vertical', size_hint_x=0.65, spacing=dp(5))

        # Waveform visualization
        self.viz_widget = Widget(size_hint_y=0.7)
        viz_area.add_widget(self.viz_widget)

        # Custom waveform editor
        custom_panel = BoxLayout(orientation='vertical', size_hint_y=0.4, spacing=dp(5))

        # Custom controls
        custom_controls = BoxLayout(size_hint_y=0.2, spacing=dp(5))
        custom_label = Label(text="Custom Waveform Editor", font_size=dp(14), color=self.WHITE)
        custom_controls.add_widget(custom_label)

        self.edit_toggle_btn = Button(text="Edit Off", font_size=dp(14), size_hint_x=0.25, background_color=self.BLUE)
        self.edit_toggle_btn.bind(on_press=self.on_edit_toggle)
        custom_controls.add_widget(self.edit_toggle_btn)

        mode_btn = Button(text=f"Mode: {self.edit_mode.capitalize()}", font_size=dp(14), size_hint_x=0.25)
        mode_btn.bind(on_press=self.on_mode_toggle)
        self.mode_btn = mode_btn
        custom_controls.add_widget(mode_btn)

        reset_btn = Button(text="Reset", font_size=dp(14), background_color=self.RED, size_hint_x=0.25)
        reset_btn.bind(on_press=self.on_reset_custom)
        custom_controls.add_widget(reset_btn)

        custom_panel.add_widget(custom_controls)

        # Edit canvas
        self.edit_widget = Widget(size_hint_y=0.8)
        custom_panel.add_widget(self.edit_widget)

        viz_area.add_widget(custom_panel)
        main.add_widget(viz_area)

        root.add_widget(main)

        # Store labels for updates
        self.freq_label = freq_label
        self.hz_step_label = hz_step_label
        self.wave_type_label = wave_type_label
        self.sweep_dur_label = sweep_dur_label
        self.sweep_start_label = sweep_start_label
        self.sweep_end_label = sweep_end_label

        # Bind touch
        self.edit_widget.bind(on_touch_down=self.on_edit_touch_down)
        self.edit_widget.bind(on_touch_move=self.on_edit_touch_move)
        self.edit_widget.bind(on_touch_up=self.on_edit_touch_up)

        return root

        # Bind keyboard events
        Window.bind(on_key_down=self.on_key_down)

        # Bind mouse events to edit widget
        self.edit_widget.bind(on_touch_down=self.on_edit_touch_down)
        self.edit_widget.bind(on_touch_move=self.on_edit_touch_move)
        self.edit_widget.bind(on_touch_up=self.on_edit_touch_up)

        # Schedule updates
        Clock.schedule_interval(self.update_display, 1/60)

        return root

    def set_volume(self, value):
        self.volume = int(value)
        if hasattr(self, 'volume_slider'):
            self.volume_slider.value = self.volume

    def toggle_sweep_loop(self):
        self.sweep_loop = not self.sweep_loop
        self.loop_btn.text = "Loop On" if self.sweep_loop else "Loop Off"

    def toggle_drawing_mode_ui(self):
        self.toggle_drawing_mode()

    def on_play_pause(self, instance):
        self.toggle_play_pause()

    def on_sweep_toggle(self, instance):
        self.toggle_sweep()

    def on_hz_down(self, instance):
        self.cycle_hz_step("down")

    def on_hz_up(self, instance):
        self.cycle_hz_step("up")

    def on_freq_down(self, instance):
        self.adjust_frequency("down")

    def on_freq_up(self, instance):
        self.adjust_frequency("up")

    def on_wave_down(self, instance):
        self.cycle_wave_type("down")

    def on_wave_up(self, instance):
        self.cycle_wave_type("up")

    def on_sweep_dur_down(self, instance):
        self.adjust_sweep_duration("down")

    def on_sweep_dur_up(self, instance):
        self.adjust_sweep_duration("up")

    def on_sweep_start_down(self, instance):
        self.adjust_sweep_start("down")

    def on_sweep_start_up(self, instance):
        self.adjust_sweep_start("up")

    def on_sweep_end_down(self, instance):
        self.adjust_sweep_end("down")

    def on_sweep_end_up(self, instance):
        self.adjust_sweep_end("up")

    def on_loop_toggle(self, instance):
        self.toggle_sweep_loop()

    def on_edit_toggle(self, instance):
        self.toggle_drawing_mode_ui()

    def on_mode_toggle(self, instance):
        self.toggle_edit_mode_ui()

    def on_reset_custom(self, instance):
        self.clear_custom_waveform()

    def on_volume_change(self, instance, touch):
        if not touch or touch.grab_current is not instance:
            self.set_volume(int(instance.value))

    def toggle_edit_mode_ui(self, btn=None):
        self.edit_mode = 'free' if self.edit_mode == 'points' else 'points'
        if hasattr(self, 'mode_btn'):
            self.mode_btn.text = f"Mode: {self.edit_mode.capitalize()}"

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if codepoint == 'q':
            self.stop()
        elif key == 32:  # space
            self.toggle_play_pause()
        elif codepoint == 's':
            self.toggle_sweep()
        elif codepoint == 'c':
            self.toggle_drawing_mode()
        elif codepoint == 'p':
            self.edit_mode = 'points'
        elif codepoint == 'f':
            self.edit_mode = 'free'
        elif codepoint == 'r':
            self.clear_custom_waveform()

    def update_display(self, dt):
        self.update_labels()
        self.update_visualization()
        self.update_edit_visualization()

    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        if not self.drawing_mode:
            self.update_custom_waveform()
        else:
            if self.edit_mode == 'free':
                self.drawing_points = []

    def clear_custom_waveform(self):
        self.custom_waveform = np.sin(np.linspace(0, 2 * np.pi, 1000))
        self.control_points = [(i / 9.0, math.sin(i / 9.0 * 2 * math.pi)) for i in range(10)]
        self.drawing_points = []
        self.wave_engine.set_custom_waveform(self.custom_waveform)

    def on_edit_touch_down(self, instance, touch):
        if self.drawing_mode and instance.collide_point(*touch.pos):
            if self.edit_mode == 'points':
                # Find the closest control point
                min_dist = float('inf')
                closest = None
                for i, (nx, ny) in enumerate(self.control_points):
                    sx = instance.x + nx * instance.width
                    sy = instance.y + instance.height / 2 + ny * instance.height / 3
                    dist = math.hypot(touch.x - sx, touch.y - sy)
                    if dist < 20 and dist < min_dist:
                        min_dist = dist
                        closest = i
                self.dragging_point = closest
            elif self.edit_mode == 'free':
                self.drawing_points = [(touch.x - instance.x, touch.y - instance.y)]

    def on_edit_touch_move(self, instance, touch):
        if self.drawing_mode and instance.collide_point(*touch.pos):
            if self.edit_mode == 'points' and self.dragging_point is not None:
                nx = self.control_points[self.dragging_point][0]
                ny = (touch.y - instance.y - instance.height / 2) / (instance.height / 3)
                ny = max(-1, min(1, ny))
                self.control_points[self.dragging_point] = (nx, ny)
                self.update_custom_waveform()
            elif self.edit_mode == 'free':
                self.drawing_points.append((touch.x - instance.x, touch.y - instance.y))

    def on_edit_touch_up(self, instance, touch):
        if self.drawing_mode:
            if self.edit_mode == 'points':
                self.dragging_point = None
            elif self.edit_mode == 'free':
                self.update_custom_waveform()

    def update_custom_waveform(self):
        if self.edit_mode == 'points':
            x_vals = np.array([p[0] for p in self.control_points])
            y_vals = np.array([p[1] for p in self.control_points])
            x_new = np.linspace(0, 1, 1000)
            self.custom_waveform = np.interp(x_new, x_vals, y_vals)
        elif self.edit_mode == 'free' and len(self.drawing_points) > 1:
            x_coords = [p[0] for p in self.drawing_points]
            y_coords = [p[1] for p in self.drawing_points]
            min_x = min(x_coords)
            max_x = max(x_coords)
            if max_x > min_x:
                x_vals = np.array([(x - min_x) / (max_x - min_x) for x in x_coords])
                y_vals = np.array([2 * (y / self.edit_widget.height) - 1 for y in y_coords])
                sort_idx = np.argsort(x_vals)
                x_vals = x_vals[sort_idx]
                y_vals = y_vals[sort_idx]
                x_new = np.linspace(0, 1, 1000)
                self.custom_waveform = np.interp(x_new, x_vals, y_vals)
            else:
                self.custom_waveform = np.sin(np.linspace(0, 2 * np.pi, 1000))
        self.wave_engine.set_custom_waveform(self.custom_waveform)

    def update_labels(self):
        is_anything_playing = self.is_playing or self.sweep_active
        self.status_label.text = f"Status: {'Playing' if is_anything_playing else 'Paused'}"
        self.status_label.color = self.GREEN if is_anything_playing else self.RED

        # Update control labels
        if hasattr(self, 'freq_label'):
            if self.frequency >= 1000:
                freq_display = f"{self.frequency/1000:.1f} kHz"
            else:
                freq_display = f"{self.frequency:.1f} Hz"
            self.freq_label.text = f"Frequency: {freq_display}"

        if hasattr(self, 'hz_step_label'):
            self.hz_step_label.text = f"Hz Step: {self.current_hz_step}"

        if hasattr(self, 'wave_type_label'):
            self.wave_type_label.text = f"Wave Type: {self.wave_type.capitalize()}"

        if hasattr(self, 'sweep_dur_label'):
            self.sweep_dur_label.text = f"Sweep Duration: {self.sweep_durations[self.current_sweep_duration_index]}s"

        if hasattr(self, 'sweep_start_label'):
            if self.sweep_start >= 1000:
                start_display = f"{self.sweep_start/1000:.1f} kHz"
            else:
                start_display = f"{self.sweep_start:.1f} Hz"
            self.sweep_start_label.text = f"Sweep Start: {start_display}"

        if hasattr(self, 'sweep_end_label'):
            if self.sweep_end >= 1000:
                end_display = f"{self.sweep_end/1000:.1f} kHz"
            else:
                end_display = f"{self.sweep_end:.1f} Hz"
            self.sweep_end_label.text = f"Sweep End: {end_display}"

    def update_visualization(self):
        self.viz_widget.canvas.clear()
        with self.viz_widget.canvas:
            # Draw wave
            Color(*self.GREEN if (self.is_playing or self.sweep_active) else self.RED)
            frequency = (
                self.sweep_current_frequency
                if self.sweep_active and self.sweep_current_frequency > 0
                else self.frequency
            )
            points = []
            for x in range(int(self.viz_widget.width)):
                wave_x = x / self.viz_widget.width
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
                    normalized_phase = phase % (2 * math.pi)
                    index = int(
                        normalized_phase / (2 * math.pi) * (len(self.custom_waveform) - 1)
                    )
                    wave_y = self.custom_waveform[index]
                wave_y *= self.volume / 100.0
                screen_x = self.viz_widget.x + x
                screen_y = (
                    self.viz_widget.y
                    + self.viz_widget.height // 2
                    - int(wave_y * self.viz_widget.height // 3)
                )
                points.extend([screen_x, screen_y])
            if len(points) > 2:
                Line(points=points, width=4)

            # Draw control points if in drawing mode
            if self.drawing_mode:
                Color(0, 1, 0, 1)  # green
                for nx, ny in self.control_points:
                    sx = self.viz_widget.x + nx * self.viz_widget.width
                    sy = self.viz_widget.y + self.viz_widget.height / 2 + ny * self.viz_widget.height / 3
                    Ellipse(pos=(sx - 5, sy - 5), size=(10, 10))

        # Update animation offset
        current_time = time.time()
        self.wave_animation_offset += (current_time - self.last_frame_time) * 200
        self.last_frame_time = current_time

    def update_edit_visualization(self):
        self.edit_widget.canvas.clear()
        if self.drawing_mode:
            with self.edit_widget.canvas:
                if self.edit_mode == 'points':
                    Color(0, 1, 0, 1)  # green
                    for nx, ny in self.control_points:
                        sx = nx * self.edit_widget.width
                        sy = self.edit_widget.height / 2 + ny * self.edit_widget.height / 3
                        Ellipse(pos=(sx - 5, sy - 5), size=(10, 10))
                elif self.edit_mode == 'free':
                    Color(1, 0, 0, 1)  # red
                    if len(self.drawing_points) > 1:
                        points = []
                        for px, py in self.drawing_points:
                            points.extend([px, py])
                        Line(points=points, width=4)

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
            Line(points=points, width=4)

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




if __name__ == "__main__":
    WaveGenerator().run()
