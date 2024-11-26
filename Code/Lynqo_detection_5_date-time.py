# Import necessary libraries
import tkinter as tk
from tkinter import messagebox, ttk  # GUI libraries
from PIL import Image, ImageTk  # For image handling and display
import numpy as np  # For numerical operations
from tflite_runtime.interpreter import Interpreter  # TensorFlow Lite interpreter for object detection model
from picamera2 import Picamera2  # To interface with the Pi camera
import cv2  # OpenCV for image manipulation and object detection bounding boxes
import json  # To handle JSON operations (for sets of items to detect)
import threading  # For parallel tasks
from time import sleep
from pynput import mouse, keyboard  # For detecting mouse and keyboard events (used for sleep mode)
import subprocess  # To run system commands
import os  # To handle file and directory operations
import tkinter.simpledialog  # To take user input via dialogs
from datetime import datetime  # Add this at the top of your code


class CameraApp:
    def __init__(self, window):
        # Set up main window properties
        self.window = window
        self.window.title("Object Detection Camera")
        self.window.configure(bg='white')
        self.window.attributes('-fullscreen', True)  # Fullscreen mode
        self.window.bind("<Escape>", self.on_closing)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # When closing, trigger 'on_closing' method

        # Top bar with date and time
        self.top_frame = tk.Frame(window, bg='white')
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Date label (initial value, will be updated soon)
        self.date_label = tk.Label(self.top_frame, text="", font=("Helvetica", 12), bg="white")
        self.date_label.pack(side=tk.LEFT, padx=20)

        # Time label (initial value, will be updated soon)
        self.time_label = tk.Label(self.top_frame, text="", font=("Helvetica", 12), bg="white")
        self.time_label.pack(side=tk.RIGHT, padx=20)

        # Start updating the date and time
        self.update_datetime()

        # Main content frame
        self.main_frame = tk.Frame(window, bg='white')
        self.main_frame.pack(expand=True)
        
        # Load sets from 'sets.json' (the list of items to detect)
        self.sets = self.load_sets('sets.json')

        # Dropdown menu for selecting a set to detect
        self.set_selection = tk.StringVar(value="<select set>")
        self.dropdown = ttk.Combobox(self.main_frame, textvariable=self.set_selection, values=self.get_sets_with_status())
        self.dropdown.pack(pady=20)  # Spacing for aesthetics

        # Start detection button
        self.start_button = tk.Button(self.main_frame, text="Start Object Detection", command=self.start_detection)
        self.start_button.pack(pady=20)

        # Quit button
        self.quit_button = tk.Button(self.main_frame, text="Quit", command=self.on_closing, bg="red", fg="white", font=("Helvetica", 16))
        self.quit_button.pack(side=tk.BOTTOM, pady=20)

        # Sleep mode button
        self.sleep_button = tk.Button(self.main_frame, text="Sleep", command=self.sleep_mode)
        self.sleep_button.pack(side=tk.BOTTOM, pady=20)

        # Label to display camera feed
        self.label = tk.Label(self.main_frame, bg="white")
        self.label.pack()

        # Status label to show detected or not-detected items
        self.status_label = tk.Label(self.main_frame, text="", bg="white", font=("Helvetica", 20))
        self.status_label.pack(pady=20)

        # Load object detection model and labels
        self.model_path = 'Code/ssd_mobilenet_v2_coco.tflite'  # Path to TFLite model
        self.labels_path = 'Code/mscoco_label_map.txt'  # Path to label names for COCO dataset
        self.labels = self.load_labels(self.labels_path)  # Load the labels into memory
        self.interpreter = Interpreter(model_path=self.model_path)  # Load model interpreter
        self.interpreter.allocate_tensors()  # Allocate memory for the model

        # Get input and output details from the model
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Set to store items to detect and detected items
        self.target_items = set()
        self.detected_items = set()

        # Camera and control variables
        self.camera = None  # Initialize camera as None
        self.paused = False  # Track if the camera feed is paused
        self.resuming = False  # Track if the camera feed is resuming after pause

        # Mouse and keyboard listeners for sleep mode
        self.mouse_listener = None
        self.keyboard_listener = None
        self.sleeping = False  # Track sleep mode status

    # Load sets of items to detect from JSON
    def load_sets(self, path):
        with open(path, 'r') as f:
            return json.load(f)  # Return a dictionary of sets

    # Get list of sets with their status (complete/incomplete)
    def get_sets_with_status(self):
        return [f"{name} ({data['status']})" for name, data in self.sets.items()]

    # Load object labels from a text file
    def load_labels(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]  # Return a list of labels

    # Start object detection process
    def start_detection(self):
        selected_set = self.set_selection.get()
        if selected_set == "<select set>":
            messagebox.showwarning("Select Set", "Please select a set to detect.")
            return

        set_name = selected_set.split(' (')[0]
        self.target_items = set(self.sets[set_name]["items"])  # Load target items for the selected set
        self.detected_items.clear()  # Reset detected items

        # Hide the dropdown and start button when detection begins
        self.dropdown.pack_forget()
        self.start_button.pack_forget()
        self.status_label.config(text="Detecting items...")  # Update status label

        # If not resuming from pause, add Close and Pause buttons
        if not self.resuming:
            self.close_button = tk.Button(self.main_frame, text="Close Camera", command=self.close_camera)
            self.close_button.pack(side=tk.BOTTOM, pady=20)

            self.incomplete_button = tk.Button(self.main_frame, text="Mark set incomplete", command=self.mark_incomplete)
            self.incomplete_button.pack(side=tk.BOTTOM, pady=20)

            self.pause_button = tk.Button(self.main_frame, text="Pause Camera", command=self.toggle_pause)
            self.pause_button.pack(side=tk.BOTTOM, pady=20)
        else:
            self.resuming = False

        # Initialize and start the camera if not already started
        if self.camera is None:
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
            self.camera.start()

        self.detect_objects()  # Start detecting objects

    # Object detection process
    def detect_objects(self):
        if self.camera:
            if not self.paused:  # Check if the camera is not paused
                image = self.camera.capture_array()  # Capture image from camera
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PIL

                # Resize image to match the model input size
                image_resized = Image.fromarray(image_rgb).resize((self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
                input_data = np.expand_dims(image_resized, axis=0)  # Prepare input for model

                # Set the input tensor and invoke the interpreter to run the model
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()

                # Extract results: bounding boxes, class indices, and confidence scores
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

                # Iterate over detected objects
                for i in range(len(scores)):
                    if scores[i] > 0.5:  # Confidence threshold
                        if int(classes[i]) < len(self.labels):
                            object_name = self.labels[int(classes[i])]  # Get object name from class index

                            # If the detected object is in the target set, add it to detected items
                            if object_name in self.target_items:
                                self.detected_items.add(object_name)

                            # Draw bounding box and label on the image
                            box = boxes[i]
                            ymin, xmin, ymax, xmax = box
                            x1, y1, x2, y2 = int(xmin * 640), int(ymin * 480), int(xmax * 640), int(ymax * 480)
                            image_rgb = cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            image_rgb = cv2.putText(image_rgb, object_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Convert the processed image back to a format suitable for Tkinter
                image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
                self.label.config(image=image_tk)
                self.label.image = image_tk  # Keep a reference to avoid garbage collection

                self.update_status()  # Update detection status on the screen

                self.window.after(10, self.detect_objects)  # Continue detecting

    # Save the current camera frame as an image
    def save_image(self):
        if not os.path.exists('Saved Images'):  # Create directory if not exists
            os.makedirs('Saved Images')

        # Capture current frame
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for saving

        # Prompt user for image name
        file_name = tk.simpledialog.askstring("Save Image", "Enter a name for the image:")
        if not file_name:
            messagebox.showwarning("No Name Entered", "No name entered. Image not saved.")
            return

        if not file_name.endswith('.png'):  # Ensure the file name ends with .png
            file_name += '.png'

        image_path = os.path.join('Saved Images', file_name)

        # Confirm overwrite if file already exists
        if os.path.exists(image_path):
            overwrite = messagebox.askyesno("Overwrite?", f"The file '{file_name}' already exists. Do you want to overwrite it?")
            if not overwrite:
                messagebox.showinfo("Save Cancelled", "Image not saved.")
                return

        # Save image to file
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Image Saved", f"Image saved to {image_path}")

    # Update status label with detected/not-detected items
    def update_status(self):
        status_text = ""
        all_detected = True

        # Check each target item for detection
        for item in self.target_items:
            if item in self.detected_items:
                status_text += f"{item}: Detected\n"
            else:
                status_text += f"{item}: Not Detected\n"
                all_detected = False

        self.status_label.config(text=status_text)

        # If all items are detected, mark set as complete
        if all_detected:
            messagebox.showinfo("Set Complete", "All target items detected!")
            self.sets[self.set_selection.get().split(' (')[0]]["status"] = "complete"
            self.save_sets()  # Save updated set status
            self.close_camera()  # Close the camera when done

    # Mark the current set as incomplete
    def mark_incomplete(self):
        self.sets[self.set_selection.get().split(' (')[0]]["status"] = "incomplete"
        self.save_sets()
        self.close_camera()

    # Save the sets (with their status) to JSON
    def save_sets(self):
        with open('sets.json', 'w') as f:
            json.dump(self.sets, f, indent=4)

    # Close the camera and reset the UI to the start screen
    def close_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None
        self.reset_to_start_screen()

    # Pause and resume detection by saving a snapshot and resuming
    def toggle_pause(self):
        self.display_raw_image()  # Display raw image while pausing
        self.paused = True    
        self.save_image()  # Save image during pause
        self.paused = False
        self.resuming = True
        self.start_detection()  # Resume detection

    # Display raw image without bounding boxes (used when pausing)
    def display_raw_image(self):
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
        self.label.config(image=image_tk)
        self.label.image = image_tk

    # Reset the UI to initial state
    def reset_to_start_screen(self):
        self.label.config(image='')
        self.label.image = None
        self.status_label.config(text="")
        self.dropdown['values'] = self.get_sets_with_status()
        self.set_selection.set("<select set>")
        self.dropdown.pack(pady=20)
        self.start_button.pack(pady=20)
        if hasattr(self, 'close_button'):
            self.close_button.pack_forget()
        if hasattr(self, 'incomplete_button'):
            self.incomplete_button.pack_forget()
        if hasattr(self, 'pause_button'):
            self.pause_button.pack_forget()
        self.paused = False

    # Handle application close
    def on_closing(self, event=None):
        if self.camera:
            self.camera.stop()
            self.camera.close()
        self.window.destroy()  # Destroy window and exit

    # Enter sleep mode: disable display and wait for keyboard/mouse input to wake up
    def sleep_mode(self):
        print("Entering sleep mode...")
        self.sleeping = True

        # Turn off the screen (depends on display configuration)
        os.system("wlr-randr --output HDMI-A-1 --off")

        # Start mouse and keyboard listeners to detect wake-up event
        mouse_listener = mouse.Listener(on_move=self.wake_up, on_click=self.wake_up)
        keyboard_listener = keyboard.Listener(on_press=self.wake_up)

        mouse_listener.start()
        keyboard_listener.start()

    # Wake up from sleep mode: re-enable display and stop listeners
    def wake_up(self, *args):
        print("Waking up from sleep mode...")
        self.sleeping = False

        # Turn the screen back on
        os.system("wlr-randr --output HDMI-A-1 --on")

        # Stop mouse and keyboard listeners
        if self.mouse_listener is not None:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

        return False  # Stop listeners
        
    def update_datetime(self):
        # Get the current date and time
        now = datetime.now()
        current_date = now.strftime("%d/%m/%Y")
        current_time = now.strftime("%H:%M:%S")
        
        # Update the labels with the current date and time
        self.date_label.config(text=current_date)
        self.time_label.config(text=current_time)
        
        # Call this method again after 1000 milliseconds (1 second)
        self.window.after(1000, self.update_datetime)

# Main function: Initialize Tkinter window and run the CameraApp
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()  # Start the GUI event loop
