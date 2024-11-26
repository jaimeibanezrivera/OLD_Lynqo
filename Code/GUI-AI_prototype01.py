from picamera2 import Picamera2, Preview
import os
import tkinter as tk
from time import sleep, time, strftime
from PIL import Image, ImageDraw, ImageFont
import random
from pynput import mouse, keyboard
import inference
from inference_sdk import InferenceHTTPClient

class CameraApp:
    def __init__(self, root, base_directory):
        self.root = root
        self.root.title("Camera App")

        #Story the base directory
        self.base_directory = base_directory
        
        # Full screen
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        self.picam2 = Picamera2()
        
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        # Stop Python button (top right corner)
        self.stop_python_button = tk.Button(root, text="Stop Python", command=self.on_closing, width=20, height=2)
        self.stop_python_button.grid(row=0, column=2, padx=20, pady=20, sticky="ne")

        # Camera button frame (center)
        self.camera_button = tk.Button(root, text="Open Camera", command=self.open_camera, width=20, height=2)
        self.camera_button.grid(row=1, column=1, padx=20, pady=20)
        
        # Stop Camera button (below Object A and B, center)
        self.stop_camera_button = tk.Button(root, text="Stop Camera", command=self.stop_camera, width=20, height=2)
        self.stop_camera_button.grid(row=3, column=1, padx=20, pady=20)
        self.stop_camera_button.grid_remove()
        
        # Sleep button
        self.sleep_button = tk.Button(root, text="Sleep", command=self.sleep_mode, width=20, height=2)
        self.sleep_button.grid(row=4, column=1, padx=20, pady=20)
        
        # Framerate label
        self.framerate_label = tk.Label(root, text="Framerate: N/A", font=("Helvetica", 16))
        self.framerate_label.grid(row=0, column=0, padx=20, pady=20)
        self.framerate_label.grid_remove()

        # Save Frame buttons
        self.save_button_a = tk.Button(root, text="Object A", command=lambda: self.save_frame("Object_A"), width=20, height=2)
        self.save_button_a.grid(row=2, column=0, padx=20, pady=20)
        self.save_button_a.grid_remove()

        self.save_button_b = tk.Button(root, text="Object B", command=lambda: self.save_frame("Object_B"), width=20, height=2)
        self.save_button_b.grid(row=2, column=2, padx=20, pady=20)
        self.save_button_b.grid_remove()

        self.mouse_listener = None
        self.keyboard_listener = None
        self.sleeping = False
        self.running = False
    
    def open_camera(self):
        try:
            print("Starting preview")
            self.picam2.start_preview(Preview.QTGL)
            self.picam2.start()
            self.running = True
            self.update_framerate()
            self.camera_button.grid_remove()

            # Show extra buttons when camera starts
            self.save_button_a.grid()
            self.save_button_b.grid()
            self.stop_camera_button.grid()
            self.framerate_label.grid()

        except Exception as e:
            print(f"Error: {e}")
    
    def update_framerate(self):
        if self.running:
            frame_info = self.picam2.capture_metadata()
            if frame_info and 'FrameDuration' in frame_info:
                frame_duration = frame_info['FrameDuration'] / 1e6  # Convert to seconds
                framerate = round(1.0 / frame_duration, 2) if frame_duration > 0 else 0
                self.framerate_label.config(text=f"Framerate: {framerate} FPS")
            else:
                self.framerate_label.config(text="Framerate: N/A")
            
            # Schedule next update
            self.root.after(1000, self.update_framerate)  # Update every second
    
    def save_frame(self, object_name):
        try:
            timestamp = strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(self.base_directory, object_name)  # Folder name based on the object name
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            filename = f"{object_name}_{timestamp}.jpg"
            filepath = os.path.join(directory, filename)
            print(f"Saving frame to {filepath}")
            
            # Capture the image
            self.picam2.capture_file(filepath)
            
            # Add accuracy parameter and display on the image
            accuracy = round(random.uniform(0, 1), 2)  # Generate a random accuracy value between 0 and 1
            self.add_accuracy_to_image(filepath, accuracy)
            
            print(f"Frame saved successfully with accuracy: {accuracy}.")
        except Exception as e:
            print(f"Error saving frame: {e}")
    
    def add_accuracy_to_image(self, filepath, accuracy):
        # Load the image
        image = Image.open(filepath)
        draw = ImageDraw.Draw(image)
        
        # Set the font and size (adjust font size/path as needed)
        font = ImageFont.load_default()
        text = f"Accuracy: {accuracy:.2f}"
        
        # Get text size and position
        text_size = draw.textsize(text, font)
        text_position = (image.width - text_size[0] - 10, image.height - text_size[1] - 10)
        
        # Add text to image
        draw.text(text_position, text, font=font, fill="white")
        
        # Save the modified image
        image.save(filepath)

    def on_closing(self):
        if self.running:
            print("Closing camera")
            self.stop_camera()
        print("Stopping Python program")
        self.root.destroy()
    
    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)
    
    def sleep_mode(self):
        print("Entering sleep mode...")
        self.sleeping = True
        self.running = False
        
        # Blank the screen
        os.system("vcgencmd display_power 0")
        
        # Start listeners for mouse and keyboard
        self.mouse_listener = mouse.Listener(on_move=self.wake_up, on_click=self.wake_up)
        self.keyboard_listener = keyboard.Listener(on_press=self.wake_up)
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
    
    def wake_up(self, *args):
        print("Waking up from sleep mode...")
        self.sleeping = False
        self.running = True
        
        # Unblank the screen
        os.system("vcgencmd display_power 1")
        
        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        self.update_framerate()  # Resume framerate updates
        
        return False  # Return False to stop the listener loop
    
    def stop_camera(self):
        try:
            if self.running:
                print("Stopping camera")
                self.picam2.stop_preview()
                self.picam2.stop()
                self.running = False

                # Hide the buttons when the camera stops
                self.save_button_a.grid_remove()
                self.save_button_b.grid_remove()
                self.stop_camera_button.grid_remove()
                self.framerate_label.grid_remove()
            else:
                print("Camera is not running")
        except Exception as e:
            print(f"Error: {e}")

    def stop_python(self):
        print("Stopping Python program")
        self.on_closing()
    
if __name__ == "__main__":
    #Specify the base directory for the images/output
    base_directory = "/home/pi/Documents/LYNQO-prototype/AI-outputs"
    
    model = InferenceHTTPClient(api_url="https://detect.roboflow.com",api_key="6SEBLtTLjfw8g1ltcCd2")
    print(f"model loaded")
    # Ensure model was successfully loaded
    if model is None:
        raise RuntimeError("Failed to load Roboflow model. Check API key, project ID, and version.")
    
    root = tk.Tk()
    app = CameraApp(root, base_directory)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
