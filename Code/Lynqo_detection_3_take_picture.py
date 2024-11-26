import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import cv2  # Make sure to import cv2 for OpenCV functions
import json
import threading
from time import sleep
from pynput import mouse, keyboard
import subprocess
import os
import tkinter.simpledialog

class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Object Detection Camera")
        self.window.attributes('-fullscreen', True)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load sets from sets.json
        self.sets = self.load_sets('sets.json')

        # Create dropdown menu for selecting a set
        self.set_selection = tk.StringVar(value="<select set>")
        self.dropdown = ttk.Combobox(window, textvariable=self.set_selection, values=self.get_sets_with_status())
        self.dropdown.pack(pady=20)

        self.start_button = tk.Button(window, text="Start Object Detection", command=self.start_detection)
        self.start_button.pack(pady=20)

        self.quit_button = tk.Button(window, text="Quit", command=self.on_closing)
        self.quit_button.pack(side=tk.BOTTOM, pady=20)
        
        self.sleep_button = tk.Button(window, text="Sleep", command=self.sleep_mode)
        self.sleep_button.pack(side=tk.BOTTOM, pady=20)

        self.label = tk.Label(window)
        self.label.pack()

        self.status_label = tk.Label(window, text="", font=("Helvetica", 20))
        self.status_label.pack(pady=20)

        self.model_path = 'Code/ssd_mobilenet_v2_coco.tflite'
        self.labels_path = 'Code/mscoco_label_map.txt'
        self.labels = self.load_labels(self.labels_path)
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.target_items = set()  # Define the set of target items
        self.detected_items = set()

        self.camera = None
        self.paused = False  # Track whether the camera feed is paused
        self.resuming = False # Track if resume button pushed
        
        self.mouse_listener = None
        self.keyboard_listener = None
        self.sleeping = False

    def load_sets(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def get_sets_with_status(self):
        return [f"{name} ({data['status']})" for name, data in self.sets.items()]

    def load_labels(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def start_detection(self):
        selected_set = self.set_selection.get()
        if selected_set == "<select set>":
            messagebox.showwarning("Select Set", "Please select a set to detect.")
            return

        set_name = selected_set.split(' (')[0]
        self.target_items = set(self.sets[set_name]["items"])
        self.detected_items.clear()

        self.dropdown.pack_forget()
        self.start_button.pack_forget()
        self.status_label.config(text="Detecting items...")
        
        if not self.resuming:
            self.close_button = tk.Button(self.window, text="Close Camera", command=self.close_camera)
            self.close_button.pack(side=tk.BOTTOM, pady=20)
            
            self.incomplete_button = tk.Button(self.window, text="Mark set incomplete", command=self.mark_incomplete)
            self.incomplete_button.pack(side=tk.BOTTOM, pady=20)

            self.pause_button = tk.Button(self.window, text="Pause Camera", command=self.toggle_pause)  # Pause button
            self.pause_button.pack(side=tk.BOTTOM, pady=20)
        else:
            self.resuming = False

        if self.camera is None:
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
            self.camera.start()

        self.detect_objects()

    def detect_objects(self):
        if self.camera:
            if not self.paused:  # Check if the camera is not paused
                image = self.camera.capture_array()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image_resized = Image.fromarray(image_rgb).resize((self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
                input_data = np.expand_dims(image_resized, axis=0)

                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()

                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

                for i in range(len(scores)):
                    if scores[i] > 0.5:  # Confidence threshold
                        if int(classes[i]) < len(self.labels):
                            object_name = self.labels[int(classes[i])]  # Look up object name from "labels" array using class index
                            if object_name in self.target_items:
                                self.detected_items.add(object_name)

                            box = boxes[i]
                            ymin, xmin, ymax, xmax = box

                            x1, y1, x2, y2 = int(xmin * 640), int(ymin * 480), int(xmax * 640), int(ymax * 480)
                            image_rgb = cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            image_rgb = cv2.putText(image_rgb, object_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
                self.label.config(image=image_tk)
                self.label.image = image_tk
                
                self.update_status()

                self.window.after(10, self.detect_objects)

    def save_image(self):
        if not os.path.exists('Saved Images'):
            os.makedirs('Saved Images')

        # Capture the current frame
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        file_name = tk.simpledialog.askstring("Save Image", "Enter a name for the image:")

        if not file_name:
            messagebox.showwarning("No Name Entered", "No name entered. Image not saved.")
            return

        if not file_name.endswith('.png'):
            file_name += '.png'

        image_path = os.path.join('Saved Images', file_name)

        if os.path.exists(image_path):
            overwrite = messagebox.askyesno("Overwrite?", f"The file '{file_name}' already exists. Do you want to overwrite it?")
            if not overwrite:
                messagebox.showinfo("Save Cancelled", "Image not saved.")
                return

        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Image Saved", f"Image saved to {image_path}")
        
        
    def update_status(self):
        status_text = ""
        all_detected = True

        for item in self.target_items:
            if item in self.detected_items:
                status_text += f"{item}: Detected\n"
            else:
                status_text += f"{item}: Not Detected\n"
                all_detected = False

        self.status_label.config(text=status_text)

        if all_detected:
            messagebox.showinfo("Set Complete", "All target items detected!")
            self.sets[self.set_selection.get().split(' (')[0]]["status"] = "complete"
            self.save_sets()
            self.close_camera()

    def mark_incomplete(self):
        self.sets[self.set_selection.get().split(' (')[0]]["status"] = "incomplete"
        self.save_sets()
        self.close_camera()

    def save_sets(self):
        with open('sets.json', 'w') as f:
            json.dump(self.sets, f, indent=4)

    def close_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None
        self.reset_to_start_screen()
        
    '''def close_camera_pause(self):
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None'''
        
    def toggle_pause(self):
        # pause the image while saving, then continue again
        '''if self.paused:
            self.paused = False
            self.resuming = True
            #self.pause_button.config(text="Pause Camera")
            self.start_detection()
        else:
            self.paused = True
            #self.pause_button.config(text="Resume Camera")
            self.save_image()
            #self.close_camera_pause()
        '''
        self.display_raw_image()
        
        self.paused = True    
        self.save_image()
        self.paused = False
        self.resuming = True
        self.start_detection()
        
    def display_raw_image(self):
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
        self.label.config(image=image_tk)
        self.label.image = image_tk


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

    def on_closing(self):
        if self.camera:
            self.camera.stop()
            self.camera.close()
        self.window.destroy()
        
    def sleep_mode(self):
        print("Entering sleep mode...")
        self.sleeping = True
        
        #blank the screen
        os.system("wlr-randr --output HDMI-A-1 --off")
        
        #start listeners for mouse and keyboard
        mouse_listener = mouse.Listener(on_move=self.wake_up, on_click=self.wake_up)
        keyboard_listener = keyboard.Listener(on_press=self.wake_up)
        
        mouse_listener.start()
        keyboard_listener.start()
        
        
    def wake_up(self, *args):
        print("Waking up from sleep mode...")
        self.sleeping = False
        
        #unblank the screen
        os.system("wlr-randr --output HDMI-A-1 --on")
        
        #stop listeners
        if self.mouse_listener is not None:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        #stop listeners (this is done by breaking the loop, causing join() to end)
        return False

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()