#!/usr/bin/env python3
import os
import time
import signal
import threading
from typing import Tuple
import speech_recognition as sr
import pyttsx3
import platform
import pyaudio
import math
import struct
import array
from prompt2sand import Prompt2Sand

p2s = Prompt2Sand()
cleanup_running = False

# Platform detection
IS_RPI = platform.system() == "Linux" and os.path.exists('/proc/device-tree/model') and 'raspberry pi' in open('/proc/device-tree/model').read().lower()

# Optional RPI-specific imports
if IS_RPI:
    import RPi.GPIO as GPIO
    from apa102 import APA102
    import gpiozero
    
    # ReSpeaker 2-Mic Hat configuration
    BUTTON_PIN = 17  # GPIO pin for the button on ReSpeaker
    LEDS_GPIO = 12   # GPIO pin for LED power

    # LED Settings
    NUM_LEDS = 3
    LED_BRIGHTNESS = 8  # Max brightness is 31

    # LED numbers
    POWER_LED = 0
    CPU_LED = 1
    LISTENING_LED = 2

    # LED colors as RGB tuples
    LED_OFF = (0, 0, 0)
    LED_RED = (255, 0, 0)
    LED_GREEN = (0, 255, 0)
    LED_BLUE = (0, 0, 255)
    LED_ORANGE = (255, 165, 0)

    # Startup animation sequence
    STARTUP_SEQUENCE = [
        (POWER_LED, LED_RED),
        (CPU_LED, LED_ORANGE),
        (LISTENING_LED, LED_BLUE)
    ]

# Settings
PATTERNS_DIR = "patterns"

# Global variables
running = True
is_recording = False
startup_animation_running = True
cancel_recording = False
last_cancel_time = 0
audio_source = None  # Will hold our persistent audio source

# Custom Recognizer class that can be interrupted
class CancellableRecognizer(sr.Recognizer):
    def __init__(self):
        super().__init__()
        self.should_cancel = False
        self.original_timeout = None
    
    def listen(self, source, timeout=None, phrase_time_limit=None, snowboy_configuration=None):
        """
        Override the listen method to check for cancellation
        """
        self.should_cancel = False
        # Store the original timeout to restore it if needed
        self.original_timeout = self.operation_timeout
        
        # Create a thread to check for cancellation
        def check_cancel():
            while not self.should_cancel and is_recording:
                if cancel_recording:
                    print("Cancel detected, forcing timeout...")
                    self.should_cancel = True
                    # Force timeout to exit listen operation, but only when cancelling
                    self.operation_timeout = 0.1
                time.sleep(0.001)  # Check very frequently (1ms)
        
        cancel_thread = threading.Thread(target=check_cancel)
        cancel_thread.daemon = True
        cancel_thread.start()
        
        try:
            # Use a longer timeout for normal speech recognition
            result = super().listen(source, timeout, phrase_time_limit, snowboy_configuration)
            # Reset the timeout to its original value
            if self.original_timeout is not None:
                self.operation_timeout = self.original_timeout
            return result
        except sr.WaitTimeoutError:
            # Reset the timeout to its original value
            if self.original_timeout is not None:
                self.operation_timeout = self.original_timeout
                
            if self.should_cancel or cancel_recording:
                # This is an expected timeout due to cancellation
                raise sr.WaitTimeoutError("Listening cancelled by user")
            else:
                # This is a normal timeout
                raise
        except Exception as e:
            # Always restore the original timeout in case of exception
            if self.original_timeout is not None:
                self.operation_timeout = self.original_timeout
            raise

# Initialize recognition objects
recognizer = CancellableRecognizer()
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 300
microphone = sr.Microphone()

# Ensure transcripts directory exists
os.makedirs(PATTERNS_DIR, exist_ok=True)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 130)

def speak_text(text):
    """Use text-to-speech to speak the given text"""
    try:
        time.sleep(0.1)
        tts_engine.say(text)
        tts_engine.runAndWait()
        time.sleep(0.1)
    except Exception as e:
        print(f"Error with text-to-speech: {e}")

def speak_text_and_cleanup(text):
    speak_text(text)
    cleanup()

class LEDControlDummy:
    """Dummy LED control for non-RPI platforms"""
    def __init__(self):
        print("LED control not available - not running on Raspberry Pi")
    
    def set_color(self, led_num: int, rgb: Tuple[int, int, int]):
        pass
    
    def turn_on(self):
        pass
    
    def turn_off(self):
        pass
    
    def blink(self, led_num: int, rgb: Tuple[int, int, int], duration: int):
        pass

class LEDControlRPI:
    """LED control for RPI with ReSpeaker"""
    def __init__(self, led_brightness=None):
        try:
            print("Initializing ReSpeaker 2-Mic Hat LEDs...")
            self.led_power = gpiozero.LED(LEDS_GPIO, active_high=False)
            self.led_brightness = led_brightness if led_brightness is not None else 10
            self.led_states = [LED_OFF] * NUM_LEDS
            self.leds = None
            # Ensure LED power is on
            print(f"Turning on LED power pin {LEDS_GPIO}...")
            self.turn_on()
            print("LED initialization complete")
        except Exception as e:
            print(f"ERROR initializing LEDs: {e}")
            self.leds = None

    def set_color(self, led_num: int, rgb: Tuple[int, int, int]):
        if led_num < 0 or led_num >= NUM_LEDS:
            return
        if self.leds is None:
            print(f"WARNING: Cannot set LED color - LED controller not initialized")
            return
        try:
            # Only update if color is different
            if self.led_states[led_num] != rgb:
                self.leds.set_pixel(led_num, rgb[0], rgb[1], rgb[2])
                self.leds.show()
                self.led_states[led_num] = rgb
                #print(f"LED {led_num} set to RGB: {rgb}")
        except Exception as e:
            print(f"Error setting LED color: {e}")

    def turn_on(self):
        try:
            # First turn on power
            print("Activating LED power...")
            self.led_power.on()
            time.sleep(0.2)  # Give power time to stabilize
            
            # Then initialize APA102 driver
            print(f"Creating APA102 driver with {NUM_LEDS} LEDs, brightness: {self.led_brightness}")
            self.leds = APA102(num_led=NUM_LEDS, global_brightness=self.led_brightness)
            self.led_states = [LED_OFF] * NUM_LEDS
            
            # Turn off all LEDs initially (to reset state)
            print("Resetting all LEDs to OFF state")
            for i in range(NUM_LEDS):
                self.leds.set_pixel(i, 0, 0, 0)
            self.leds.show()
            
            print("LED power and initialization successful")
        except Exception as e:
            print(f"ERROR turning on LEDs: {e}")
            self.leds = None

    def turn_off(self):
        try:
            if self.leds is not None:
                print("Turning off all LEDs")
                for i in range(NUM_LEDS):
                    self.set_color(i, LED_OFF)
                self.leds.cleanup()
                self.leds = None
            print(f"Turning off LED power pin {LEDS_GPIO}")
            self.led_power.off()
        except Exception as e:
            print(f"Error turning off LEDs: {e}")

    def blink(self, led_num: int, rgb: Tuple[int, int, int], duration: int):
        if self.leds is None:
            print("WARNING: Cannot blink LED - LED controller not initialized")
            return
        original_color = self.led_states[led_num]
        try:
            #print(f"Blinking LED {led_num} with color {rgb} for {duration} cycles")
            for _ in range(duration):
                self.set_color(led_num, rgb)
                time.sleep(0.3)
                self.set_color(led_num, LED_OFF)
                time.sleep(0.3)
            self.set_color(led_num, original_color)
        except Exception as e:
            print(f"Error blinking LED: {e}")

# Initialize LED controller based on platform
led_control = LEDControlRPI(led_brightness=LED_BRIGHTNESS) if IS_RPI else LEDControlDummy()

def check_cancel_input():
    """Check if cancel input (button/Enter) was triggered"""
    global cancel_recording, last_cancel_time
    
    # Prevent rapid re-triggering
    current_time = time.time()
    if current_time - last_cancel_time < 0.5:  # Reduced cooldown to 0.5 seconds for more responsive cancellation
        return False
    
    if IS_RPI:
        # Check if button is pressed
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            # For cancelling, we need to ensure this is a NEW button press
            # not the same press that started the recording
            if current_time - last_cancel_time > 1.0:  # Reduced to 1 second for more responsive cancellation
                cancel_recording = True
                last_cancel_time = current_time
                print("\nCancelling recording...")
                return True
    else:
        # Check if Enter was pressed (non-blocking)
        if platform.system() == 'Windows':
            import msvcrt
            # Check if a key is available
            while msvcrt.kbhit():
                key = msvcrt.getch()
                if key in [b'\r', b'\n']:  # Both Enter and Return
                    cancel_recording = True
                    last_cancel_time = current_time
                    print("\nCancelling recording...")
                    # Clear any remaining input in the buffer
                    while msvcrt.kbhit():
                        msvcrt.getch()
                    return True
        else:
            # Unix-like systems (Linux/Mac)
            import sys
            import select
            
            # Check if input is available
            if select.select([sys.stdin], [], [], 0.005)[0]:  # Reduced timeout to 5ms for faster response
                key = sys.stdin.read(1)
                if key in ['\n', '\r']:  # Both Enter and Return
                    cancel_recording = True
                    last_cancel_time = current_time
                    print("\nCancelling recording...")
                    # Clear input buffer
                    import termios
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    return True
    
    return False

def play_beep(frequency=1000, duration=0.2, volume=0.5):
    """
    Play a beep sound using pyaudio
    
    Parameters:
    - frequency: tone frequency in Hz (higher = higher pitch)
    - duration: length of beep in seconds
    - volume: volume level from 0.0 to 1.0
    """    
    try:
        # Define the audio parameters
        p = pyaudio.PyAudio()
        
        # Audio parameters
        format = pyaudio.paFloat32  # Audio format
        channels = 1  # Number of channels (mono)
        rate = 44100  # Sample rate (Hz)
        
        # Create the audio stream
        stream = p.open(format=format, channels=channels, rate=rate, output=True)
        
        # Generate the beep sound (a sine wave)
        samples = (rate * duration)  # Number of samples to generate
        
        # Create the sine wave
        sine_wave = array.array('f', [0.0] * int(samples))
        for i in range(int(samples)):
            sine_value = math.sin(2.0 * math.pi * frequency * i / rate)
            sine_wave[i] = float(volume * sine_value)
        
        # Convert to binary data
        packed_wave = struct.pack('%df' % len(sine_wave), *sine_wave)
        
        # Play the sound
        stream.write(packed_wave)
        
        # Clean up the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"Error playing beep: {e}")

def record_and_transcribe():
    """Record audio and transcribe it using Google Speech Recognition"""
    global is_recording, cancel_recording, last_cancel_time, audio_source
    
    if IS_RPI:
        led_control.set_color(LISTENING_LED, LED_BLUE)
    
    print("Listening for speech... (Press button/Enter to cancel)")
    is_recording = True
    cancel_recording = False
    
    try:
        # Create a thread to check for cancel input
        def check_cancel_loop():
            while not cancel_recording and is_recording:
                if check_cancel_input():
                    # Immediately play a cancel beep
                    play_beep(frequency=400, duration=0.3, volume=0.3)
                    time.sleep(0.05)
                    play_beep(frequency=400, duration=0.3, volume=0.3)
                    break
                time.sleep(0.005)  # Check more frequently (5ms instead of 10ms)
        
        cancel_thread = threading.Thread(target=check_cancel_loop)
        cancel_thread.daemon = True
        cancel_thread.start()
        
        # Use the persistent audio source
        print("Listening...")
        
        # Play a "start listening" beep - higher pitch
        play_beep(frequency=1200, duration=0.3, volume=0.3)
        
        try:
            print(f"Engergy level: {recognizer.energy_threshold}")
            audio = recognizer.listen(audio_source, timeout=10, phrase_time_limit=10)
            
            # Check if cancel was requested during recording
            if cancel_recording or recognizer.should_cancel:
                print("\nRecording cancelled by user")
                if IS_RPI:
                    led_control.blink(LISTENING_LED, LED_RED, 1)
                # Set a longer cooldown after cancellation
                last_cancel_time = time.time() + 1.0  # Add extra cooldown time
                return
            
            # Play an "end listening" beep - lower pitch
            play_beep(frequency=800, duration=0.3, volume=0.3)
            
            print("Recognizing...")
            
            try:
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")

                # handle prompt cases
                draw_prompt = p2s.handle_prompt_cases(text, speak_text, speak_text_and_cleanup, IS_RPI)

                if draw_prompt:
                    print(f"Extracted drawing prompt: {draw_prompt}")
                    pattern_path = os.path.join(PATTERNS_DIR, f"{draw_prompt.replace(' ', '_')}.thr")
                    theta_rho_file = os.path.join("custom_patterns", os.path.basename(pattern_path)).replace('\\', '/')
                    theta_rho_files = p2s.list_theta_rho_files()
                    # check our list of theta_rho files. If its none or we already have a match to the theta_rho_file, skip the image generation
                    if theta_rho_files is None:
                        print(f"No theta_rho files found")
                        speak_text("Cannot reach DuneWeaver.")
                    elif any(theta_rho_file in file for file in theta_rho_files):
                        print(f"Skipping image generation for: {draw_prompt} because it already exists")
                        runResponse = p2s.run_theta_rho(theta_rho_file)
                        if "success" in runResponse and runResponse["success"]:
                            speak_text(f"Weaving the dunes for: {draw_prompt}")
                        else:
                            print(f"Error running theta_rho: {runResponse['detail']}")
                            error_message = runResponse['detail']
                            if runResponse['detail'].startswith(r'\d+:'):
                                error_code = runResponse['detail'].split(':')[0]
                                error_message = runResponse['detail'].split(':')[1]
                            speak_text(f"Sorry, I couldn't weave the dunes. {error_message}")
                    else:
                        speak_text(f"I'll generate {draw_prompt}")
                        print(f"Generating image for: {draw_prompt}")
                        
                        if IS_RPI:
                            led_control.set_color(LISTENING_LED, LED_ORANGE)
                        # Generate image using Gemini
                        image = p2s.generate_image_with_gemini(draw_prompt)
                        
                        if image:
                            # Convert image to sand pattern
                            try:
                                result = p2s.convert_image_to_sand(image)
                                
                                # Save the pattern                                    
                                with open(pattern_path, 'w') as f:
                                    f.write(result['formatted_coords'])
                                
                                print(f"Sand pattern saved to: {pattern_path}")
                                print(f"Number of points in pattern: {result['point_count']}")
                                
                                if IS_RPI:
                                    led_control.blink(LISTENING_LED, LED_GREEN, 2)
                                
                                uploadResponse = p2s.upload_theta_rho(pattern_path)
                                # check if response has a "success" key and if it's true
                                # runResponse["detail"] is the error message and may have the error code up front... (409: Another pattern is already running)
                                # but it also may not have the error code up front... (Another pattern is already running)
                                if "success" in uploadResponse and uploadResponse["success"]:
                                    theta_rho_file = os.path.join("custom_patterns", os.path.basename(pattern_path)).replace('\\', '/')
                                    runResponse = p2s.run_theta_rho(theta_rho_file)
                                    if "success" in runResponse and runResponse["success"]:
                                        speak_text(f"Weaving the dunes for: {draw_prompt}")
                                    else:
                                        print(f"Error running theta_rho: {runResponse['detail']}")
                                        error_message = runResponse['detail']
                                        if runResponse['detail'].startswith(r'\d+:'):
                                            error_code = runResponse['detail'].split(':')[0]
                                            error_message = runResponse['detail'].split(':')[1]
                                        speak_text(f"Sorry, I couldn't weave the dunes. {error_message}")
                                else:
                                    print(f"Error uploading theta_rho: {uploadResponse['detail']}")
                                    error_message = uploadResponse['detail']
                                    if uploadResponse['detail'].startswith(r'\d+:'):
                                        error_code = uploadResponse['detail'].split(':')[0]
                                        error_message = uploadResponse['detail'].split(':')[1]
                                    speak_text(f"Sorry, I couldn't upload the pattern to DuneWeaver. {error_message}")

                                
                            except Exception as e:
                                print(f"Error converting image to sand pattern: {e}")
                                if IS_RPI:
                                    led_control.blink(LISTENING_LED, LED_RED, 2)
                                speak_text("Sorry, I couldn't convert the image to a sand pattern.")
                            
                        else:
                            if IS_RPI:
                                led_control.blink(LISTENING_LED, LED_RED, 2)
                            speak_text("Sorry, I couldn't generate the image.")
                            
            except sr.UnknownValueError:
                print("Could not understand the audio")
                # Play an "error" beep - low pitch
                play_beep(frequency=300, duration=0.2, volume=0.3)
                
                if IS_RPI:
                    led_control.blink(LISTENING_LED, LED_RED, 2)
                speak_text("Sorry, I couldn't understand that.")
                
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                # Play an "error" beep - low pitch
                play_beep(frequency=300, duration=0.2, volume=0.3)
                
                if IS_RPI:
                    led_control.blink(LISTENING_LED, LED_RED, 2)
                speak_text("Sorry, there was an error with speech recognition.")

        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            # Play a "timeout" beep - two short low beeps
            play_beep(frequency=500, duration=0.3, volume=0.3)
            time.sleep(0.1)
            play_beep(frequency=500, duration=0.3, volume=0.3)
            
            if IS_RPI:
                led_control.blink(LISTENING_LED, LED_ORANGE, 1)
            speak_text("I didn't hear anything. Please try again.")
        
        except Exception as e:
            if not cancel_recording:
                print(f"Error during speech recognition: {e}")
                # Play an "error" beep - low pitch
                play_beep(frequency=300, duration=0.2, volume=0.3)
                
                if IS_RPI:
                    led_control.blink(LISTENING_LED, LED_RED, 2)
                speak_text("Sorry, an error occurred.")
    
    finally:
        # Ensure cancel flag is reset
        is_recording = False
        cancel_recording = False
        if IS_RPI:
            led_control.set_color(LISTENING_LED, LED_OFF)
        
        # Set a cooldown period to prevent immediate restart
        last_cancel_time = time.time()

def cleanup():
    """Clean up resources"""
    global running, audio_source, cleanup_running
    running = False
    if cleanup_running:
        print("Cleanup already running, skipping")
        return
    cleanup_running = True
    print("Cleaning up resources...")
    
    try:
        # Say goodbye before exiting
        speak_text("The only winning move is not to play.")
        
        # Properly close the audio source if it's open
        if audio_source is not None:
            try:
                microphone.__exit__(None, None, None)
                print("Microphone stream closed")
            except Exception as e:
                print(f"Error closing microphone stream: {e}")
        
        if IS_RPI:
            # Turn off all LEDs first
            for i in range(NUM_LEDS):
                led_control.set_color(i, LED_OFF)
            
            # Cleanup LED resources
            led_control.turn_off()
            
            # Give a small delay to ensure LED operations complete
            time.sleep(0.2)
            
            # Explicitly set GPIO mode before cleanup to avoid the error
            if not GPIO.getmode():
                GPIO.setmode(GPIO.BCM)
            
            # Clean up GPIO pins
            GPIO.cleanup()
            
            print("GPIO resources cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals"""
    global running
    print("Exiting...")
    running = False
    cleanup()
    # Exit immediately to prevent further automatic cleanup
    import os
    os._exit(0)

def startup_animation():
    """Run a continuous LED animation sequence until startup is complete."""
    global startup_animation_running
    sequence_index = 0
    
    print("Starting LED animation sequence...")
    try:
        while startup_animation_running:
            # Get the next LED and color in the sequence
            led_num, color = STARTUP_SEQUENCE[sequence_index]
            
            # Light up the LED
            #print(f"Animation: Lighting LED {led_num} with color {color}")
            led_control.set_color(led_num, color)
            time.sleep(0.2)
            led_control.set_color(led_num, LED_OFF)
            time.sleep(0.1)
            
            # Move to next LED in sequence
            sequence_index = (sequence_index + 1) % len(STARTUP_SEQUENCE)
        
        print("Startup animation completed")
    except Exception as e:
        print(f"Error in startup animation: {e}")

def main():
    global running, startup_animation_running, cancel_recording, last_cancel_time, audio_source
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Starting voice transcriber...")
        
        # Start the startup animation in a separate thread if on RPI
        startup_thread = None
        if IS_RPI:
            # Set up GPIO for button if on RPI
            print("Setting up GPIO for ReSpeaker 2-Mic Hat...")
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Wait for the button to stabilize
            print("Waiting for button to stabilize...")
            time.sleep(1)
            
            # Report initial button state
            button_state = "PRESSED" if GPIO.input(BUTTON_PIN) == GPIO.LOW else "RELEASED"
            print(f"Initial button state: {button_state}")
            
            # Start the animation
            startup_animation_running = True
            startup_thread = threading.Thread(target=startup_animation)
            startup_thread.daemon = True
            startup_thread.start()
            
            # Give the startup animation some time to run before proceeding
            time.sleep(2)
        
        # Initialize microphone and create a persistent audio source
        print("Initializing microphone and creating audio stream...")
        audio_source = microphone.__enter__()
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(audio_source, duration=2)
        print("Microphone initialized and ready")
        
        # Stop the startup animation
        print("Stopping startup animation...")
        startup_animation_running = False
        if startup_thread:
            startup_thread.join(timeout=1)
            print("Startup animation thread completed")
        
        if IS_RPI:
            print("Setting POWER_LED to GREEN to indicate ready state")
            led_control.set_color(POWER_LED, LED_GREEN)
            print(f"Running in RPI mode")
            
            # Wait for any currently pressed button to be released before starting
            print("Waiting for button to be released if it's currently pressed...")
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.1)
            print("Button is now released, ready to accept button presses")
        else:
            print("Running in desktop mode")
        
        # Set an initial cancel time in the past
        last_cancel_time = time.time() - 5.0
        
        # Play startup sound and greeting
        #time.sleep(0.5)  # Brief pause before greeting
        speak_text("Greetings Professor Fallken.")
        speak_text("Shall we weave some dunes? Press the button and wait for the beep.")
        
        while running:
            if IS_RPI:
                # On RPI, wait for button press
                if GPIO.input(BUTTON_PIN) == GPIO.LOW and not is_recording:
                    # Check if enough time has passed since last cancellation
                    current_time = time.time()
                    if current_time - last_cancel_time > 2.0:  # Increased to 2 seconds to prevent accidental restart
                        cancel_recording = False  # Reset cancel flag
                        # Record the time when we start recording
                        last_cancel_time = current_time
                        print("Button pressed - starting recording...")
                        # Add a small delay to make sure the button is released
                        # before we start checking for cancellation
                        time.sleep(0.5)
                        record_and_transcribe()
                    else:
                        # If trying to start too quickly, show message
                        print(f"Please wait a moment before starting again... ({int(2.0 - (current_time - last_cancel_time))}s)")
                        # Add a small delay to prevent button bounce
                        time.sleep(0.5)
            else:
                try:
                    # On desktop, wait for Enter key
                    input("Press Enter to start recording...")
                    if not is_recording:
                        # Check if enough time has passed since last operation
                        current_time = time.time()
                        if current_time - last_cancel_time > 2.0:  # Increased cooldown to 2 seconds
                            cancel_recording = False  # Reset cancel flag
                            last_cancel_time = current_time  # Update to prevent immediate cancellation
                            record_and_transcribe()
                        else:
                            # If trying to start too quickly after last operation, show message
                            remaining = max(0, int(2.0 - (current_time - last_cancel_time)))
                            print(f"Please wait a moment before starting again... ({remaining}s)")
                            # Add a small delay to prevent immediate retry
                            time.sleep(0.5)
                except EOFError:
                    # Handle Ctrl+D gracefully
                    running = False
            
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error: {e}")
        running = False
    finally:
        startup_animation_running = False
        if startup_thread:
            startup_thread.join(timeout=1)
        cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This should catch Ctrl+C if the signal handler somehow misses it
        print("Keyboard interrupt received")
    finally:
        # Ensure cleanup happens
        cleanup() 