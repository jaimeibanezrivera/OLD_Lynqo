import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import cv2  # Make sure to import cv2 for OpenCV functions
import json

class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Object Detection Camera")
        self.window.attributes('-fullscreen', True)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load sets from sets.json
        self.sets = self.load_sets('Code/sets.json')

        # Create dropdown menu for selecting a set
        self.set_selection = tk.StringVar(value="<select set>")
        self.dropdown = ttk.Combobox(window, textvariable=self.set_selection, values=self.get_sets_with_status())
        self.dropdown.pack(pady=20)

        self.start_button = tk.Button(window, text="Start Object Detection", command=self.start_detection)
        self.start_button.pack(pady=20)

        self.quit_button = tk.Button(window, text="Quit", command=self.on_closing)
        self.quit_button.pack(side=tk.BOTTOM, pady=20)

        self.label = tk.Label(window)
        self.label.pack()

        self.status_label = tk.Label(window, text="", font=("Helvetica", 20))
        self.status_label.pack(pady=20)

        self.model_path = 'Code/model.tflite'
        self.labels_path = 'Code/lynqo_label_map.txt'
        self.labels = self.load_labels(self.labels_path)
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.target_items = set()  # Define the set of target items
        self.detected_items = set()

        self.camera = None

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
        self.close_button = tk.Button(self.window, text="Close Camera", command=self.close_camera)
        self.close_button.pack(side=tk.BOTTOM, pady=20)
        self.incomplete_button = tk.Button(self.window, text="Mark set incomplete", command=self.mark_incomplete)
        self.incomplete_button.pack(side=tk.BOTTOM, pady=20)

        if self.camera is None:
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
            self.camera.start()

        self.detect_objects()

    def detect_objects(self):
        if self.camera:
            image = self.camera.capture_array()
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_resized = Image.fromarray(image_gray).resize((self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]))
            image_data = np.expand_dims(image_resized, axis=0)
            image_data = np.expand_dims(image_data, axis=-1)
            image_data = image_data.astype(np.float32)

            '''self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
            # Check the output details length and print to debug
            #print(f"Output details: {self.output_details}")

            # Make sure the index is valid
            #if len(self.output_details) > 1:
            #    output = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            #else:
            #    print("Output index out of range!")
            #    return
            
            self.interpreter.invoke()

            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            for i in range(np.max(x.shape)):
                if output[i] > 0.5:  # Confidence threshold
                    if int(i) < len(self.labels):
                        object_name = self.labels[int(i)]  # Look up object name from "labels" array using class index
                        if object_name in self.target_items:
                            self.detected_items.add(object_name)
            '''
            
            # Set the input tensor and invoke the interpreter to run the model
            self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
            self.interpreter.invoke()

            # Extract results: bounding boxes, class indices, and confidence scores
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # Iterate over detected objects
            for i in range(len(scores)):
                if scores[i] > 0.5:  # Confidence threshold
                    if int(i) < len(self.labels):
                        object_name = self.labels[int(i)]  # Get object name from class index

                        # If the detected object is in the target set, add it to detected items
                        #if object_name in self.target_items:
                        self.detected_items.add(object_name)
                        print(f"Detected {object_name}")
            
            image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_gray))
            self.label.config(image=image_tk)
            self.label.image = image_tk
            
            self.update_status()

            self.window.after(10, self.detect_objects)

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

    def on_closing(self):
        if self.camera:
            self.camera.stop()
            self.camera.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
