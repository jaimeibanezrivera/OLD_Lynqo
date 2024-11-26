"""
Python program that uses our own thaught Roboflow model, to recognise medical instruments
GUI is adapted, with the use of "Take picture", the AI model is triggered, and the instrument's picture is added to the correct folder
"""

import os
import tkinter as tk
from time import strftime
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
from roboflow import Roboflow
from picamera2 import Picamera2, Preview
import random
import time
from pynput import mouse, keyboard
import inference
from inference_sdk import InferenceHTTPClient
import os

class CameraApp:
    def __init__(self, root, base_directory):
        self.root = root
        self.root.title("Camera App")
        self.base_directory = base_directory
        self.model = model  # Store the model
        
        # Full screen
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        self.picam2 = Picamera2()
        
        # Configure grid layout of GUI
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        # Label to show the camera feed
        self.camera_feed_label = tk.Label(self.root)
        self.camera_feed_label.grid(row=1, column=1, padx=20, pady=20)
        # Stop Python button
        self.stop_python_button = tk.Button(root, text="Stop Python", command=self.on_closing, width=20, height=2)
        self.stop_python_button.grid(row=0, column=2, padx=20, pady=20, sticky="ne")
        # Camera button
        self.camera_button = tk.Button(root, text="Open Camera", command=self.open_camera, width=20, height=2)
        self.camera_button.grid(row=2, column=1, padx=20, pady=20)
        # Stop Camera button
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
        # Save Frame buttons    ONLY FOR DEMO PURPOSES
        self.save_button_a = tk.Button(root, text="Object A", command=lambda: self.save_frame("Object_A", 0.2), width=20, height=2)
        self.save_button_a.grid(row=2, column=0, padx=20, pady=20)
        self.save_button_a.grid_remove()
        self.save_button_b = tk.Button(root, text="Object B", command=lambda: self.save_frame("Object_B", 0.2), width=20, height=2)
        self.save_button_b.grid(row=2, column=2, padx=20, pady=20)
        self.save_button_b.grid_remove()
        # Take Picture button - TRIGGERS AI software
        self.take_picture_button = tk.Button(root, text="Take Picture", command=self.take_picture, width=20, height=2)
        self.take_picture_button.grid(row=2, column=1, padx=20, pady=20)
        self.take_picture_button.grid_remove()

        self.mouse_listener = None
        self.keyboard_listener = None
        self.sleeping = False
        self.running = False
    
    def open_camera(self):
        try:
            print("Starting preview")
            self.picam2.start_preview(Preview.NULL)
            #self.picam2.start()
            #self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
            self.picam2.start()
            self.running = True
            #self.update_framerate()
            self.camera_button.grid_remove()
            # Show extra buttons when camera starts
            self.save_button_a.grid()
            self.save_button_b.grid()
            self.stop_camera_button.grid()
            #self.framerate_label.grid()
            self.take_picture_button.grid()

        except Exception as e:
            print(f"Error: {e}")
    
    def update_camera_feed(self):
        if self.running:
            try:
                # Capture a frame from the camera
                frame = self.picam2.capture_array()
                
                # Convert the frame (numpy array) into an image
                frame_image = Image.fromarray(frame)
                frame_image_tk = ImageTk.PhotoImage(frame_image)
                
                # Update the label with the new image
                self.camera_feed_label.config(image=frame_image_tk)
                self.camera_feed_label.image = frame_image_tk  # Keep a reference to avoid garbage collection

                print("Frame updated successfully")  # Debug statement
            except Exception as e:
                print(f"Error capturing frame: {e}")
            
            # Schedule next frame update
            self.root.after(10, self.update_camera_feed)  # Update every 10 ms
            
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
    
    def save_frame(self, object_name, accuracy):
        try:
            timestamp = strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(self.base_directory, object_name)
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
        os.system("wlr-randr --output HDMI-A-1 --off")
        
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
        os.system("wlr-randr --output HDMI-A-1 --on")
        
        # Stop listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        self.update_framerate()  # Resume framerate updates
        
        return False  # Return False to stop the listener loop 
    
    def take_picture(self):
        if model is None:
            print("Roboflow model is not initialized.")
            return
        
        try:
            # Capture the current frame
            timestamp = strftime("%Y%m%d_%H%M%S")
            filename = f"Capture_{timestamp}.jpg"
            filepath = os.path.join(self.base_directory, filename)
            print(f"Taking picture and saving to {filepath}")
            self.picam2.capture_file(filepath)
            print("File captured")

            result = model.infer(filepath,model_id="medicalcomponents/1")
            predictions = result.get('predictions', []) #extract the predictions
            sorted_predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True) # Step 2: Sort the predictions by confidence in descending order
            top_three_predictions = sorted_predictions[:3] # Step 3: Retrieve the top three predictions
            for i, pred in enumerate(top_three_predictions, start=1): # Step 4: Print the top three classes and confidences
                print(f"Top {i}: Class = {pred['class']}, Confidence = {pred['confidence']}")
            self.save_frame(object_name=top_three_predictions[0]['class'], accuracy=top_three_predictions[0]['confidence'])

            """
            # Optionally, store these in variables if you need to use them later
            top1_class = top_three_predictions[0]['class']
            top1_confidence = top_three_predictions[0]['confidence']

            top2_class = top_three_predictions[1]['class']
            top2_confidence = top_three_predictions[1]['confidence']

            top3_class = top_three_predictions[2]['class']
            top3_confidence = top_three_predictions[2]['confidence']
            """
        
        except Exception as e:
            print(f"Error taking picture: {e}")

    

    def stop_camera(self):
        try:
            if self.running:
                print("Stopping camera")
                self.picam2.stop_preview()
                self.picam2.stop()
                self.running = False
                self.camera_button.grid()
                self.stop_camera_button.grid_remove()
            else:
                print("Camera is not running")
        except Exception as e:
            print(f"Error: {e}")
            
    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

    def stop_python(self):
        print("Stopping Python program")
        self.on_closing()
    
if __name__ == "__main__":
    # Specify the base directory for the images/output
    base_directory = "/home/LynqoProto/Lynqo_detection/Code/AI-outputs"

    # Initialize Roboflow with your API key
    try:

        model = InferenceHTTPClient(api_url="https://detect.roboflow.com",api_key="6SEBLtTLjfw8g1ltcCd2")

        # Ensure model was successfully loaded
        if model is None:
            raise RuntimeError("Failed to load Roboflow model. Check API key, project ID, and version.")

        else:
            root = tk.Tk()
            app = CameraApp(root, base_directory)
            root.protocol("WM_DELETE_WINDOW", app.on_closing)
            root.mainloop()

    except Exception as e:
        print(f"Error initializing Roboflow model: {e}")
        model = None
    
    
