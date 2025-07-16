import keyboard
import time

print("Key Test Script")
print("===============")
print("Press any key to see its name. Press 'q' to quit.")
print("Focus on testing the arrow keys: ↑ ↓ ← →")
print()

def on_key_event(event):
    if event.event_type == keyboard.KEY_DOWN:
        print(f"Key pressed: '{event.name}' (scan code: {event.scan_code})")
        if event.name == 'q':
            print("Exiting...")
            keyboard.unhook_all()
            exit()

keyboard.hook(on_key_event)

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nExiting...")
    keyboard.unhook_all()
