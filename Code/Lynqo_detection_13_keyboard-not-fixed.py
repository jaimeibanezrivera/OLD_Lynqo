# Import necessary libraries
import tkinter as tk
from tkinter import messagebox, ttk, font  # GUI libraries
from tkinter.font import Font
import customtkinter as ctk
from customtkinter import CTkButton, CTkFont, set_appearance_mode
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
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLineEdit, QPushButton, QLabel



class CameraApp:
    def __init__(self, window):
        # Set up main window properties
        self.window = window
        self.window.title("Object Detection Camera")
        self.window.configure(bg='white')
        self.window.attributes('-fullscreen', True)  # Fullscreen mode
        #self.window.attributes("-type", "splash")
        self.window.bind("<Escape>", self.on_closing)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # When closing, trigger 'on_closing' method

        # Ensure window updates and apply full-screen again if needed
        self.window.update_idletasks()
        self.window.after(100, lambda: self.window.attributes('-fullscreen', True))  # Reapply after delay
        
        #self.window.bind_class("Entry", "<1>", lambda ev: ev.widget.focus_force())
        #self.window.attributes('-topmost', False)
        
        # Get the screen dimensions
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        
        # Set appearance mode (optional: light, dark, or system)
        ctk.set_appearance_mode("light")
        
        # load the GUI icons
        self.load_icons()
    
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
        
        # Bottom bar with quit and
        self.bottom_frame = tk.Frame(self.window, bg='white', borderwidth=0, relief="solid")  # Add border
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Add the quit button once (permanently)
        self.add_quit_button()

        # Main content frame (using dynamic screen height for padding)
        self.main_frame = tk.Frame(window, bg='white')
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Set custom fonts
        self.button_font = CTkFont(family="Helvetica", size=14, weight="bold")
        
        
        # Camera and control variables
        self.camera = None  # Initialize camera as None
        self.paused = False  # Track if the camera feed is paused
        self.resuming = False  # Track if the camera feed is resuming after pause

        # Display the home screen after initialization
        self.Keuzemenu_OK()
        
        # Load sets from 'sets.json' (the list of items to detect)
        self.sets = self.load_sets('Code/sets.json')

        self.label = tk.Label(window, bg="white")
        self.label.pack()

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


        # Mouse and keyboard listeners for sleep mode
        self.mouse_listener = None
        self.keyboard_listener = None
        self.sleeping = False  # Track sleep mode status
        self.touch_device = None
        
        self.camera_width = 1920
        self.camera_height = 1477
        
        self.reset_set_selection()
        

    
    
    
    
    
    
    #######################################################################
    ####### GUI functions
    #######################################################################
    
    def update_datetime(self):
        # Get the current date and time
        now = datetime.now()
        current_date = now.strftime("%d/%m/%Y")
        current_time = now.strftime("%H:%M")
        
        # Update the labels with the current date and time
        self.date_label.config(text=current_date)
        self.time_label.config(text=current_time)
        
        # Call this method again after 1000 milliseconds (1 second)
        self.window.after(1000, self.update_datetime)
        
        
    # load the GUI icons
    def load_icons(self):
        # Load the home icon image
        self.home_image = Image.open("Code/Images_GUI/home.png")
        self.home_image = self.home_image.resize((35, 35), Image.LANCZOS)
        self.home_icon = ImageTk.PhotoImage(self.home_image)
        
        # Load the quit icon image
        self.quit_image = Image.open("Code/Images_GUI/quit.png")
        self.quit_image = self.quit_image.resize((35, 35), Image.LANCZOS)
        self.quit_icon = ImageTk.PhotoImage(self.quit_image)
        
        # Load the sleep icon image
        self.sleep_image = Image.open("Code/Images_GUI/sleep.png")
        self.sleep_image = self.sleep_image.resize((35, 35), Image.LANCZOS)
        self.sleep_icon = ImageTk.PhotoImage(self.sleep_image)
    
    
    # Add the quit button permanently in the bottom frame
    def add_quit_button(self):
        """Add a quit button to the bottom left of the screen."""
        self.quit_button = tk.Button(self.bottom_frame, image=self.quit_icon, command=self.on_closing, bd=0, bg='white', activebackground='white', highlightthickness=0)
        self.quit_button.pack(side=tk.LEFT, padx=20, pady=10)
        
        """Add a sleep button to the bottom center of the screen."""
        self.sleep_button = tk.Button(self.bottom_frame, image=self.sleep_icon, command=self.sleep_mode, bd=0, bg='white', activebackground='white', highlightthickness=0)
        self.sleep_button.pack(pady=10)
        self.sleep_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


    def add_home_button(self):
        """Add a home button to the bottom right of the screen."""
        # Only create the button if it doesn't already exist
        if not hasattr(self, 'home_button'):
            self.home_button = tk.Button(self.bottom_frame, image=self.home_icon, command=self.Keuzemenu_OK, bd=0, bg='white', activebackground='white', highlightthickness=0)
        self.home_button.pack(side=tk.RIGHT, padx=20, pady=10)  # Pack the button to ensure it's visible


    def remove_home_button(self):
        """Remove the home button if it exists."""
        if hasattr(self, 'home_button'):
            self.home_button.pack_forget()  # Hide the button from the screen
        
        
    def get_scaling_factor(self):
        """Return a scaling factor based on the current screen DPI."""
        # Get the DPI of the current display
        if self.window.tk.call("tk", "windowingsystem") == 'x11':
            # For X11 (Linux systems like Raspberry Pi)
            dpi = self.window.winfo_fpixels('1i')
        else:
            # Use a default DPI value if the system doesn't support DPI querying
            dpi = 96
            
        # Reference DPI for scaling (this might need tuning based on your displays)
        reference_dpi = 96
    
        # Calculate scaling factor relative to reference DPI
        scaling_factor = dpi / reference_dpi
        return scaling_factor
    
    # Page after login
    def Keuzemenu_OK(self):
        """Create the home screen with two rounded buttons."""
        self.clear_screen()
        self.remove_home_button()  # Ensure the home button is removed when this screen is active

        # Create a container frame to hold the buttons, centered in the main frame
        button_container = tk.Frame(self.main_frame, bg='white')
        button_container.grid(row=0, column=0, sticky="n")  # Ensure the container expands and fills space

        # Configure the main frame to resize properly
        self.main_frame.grid_rowconfigure(0, weight=1)  # Allow vertical resizing
        self.main_frame.grid_columnconfigure(0, weight=1)  # Allow horizontal resizing

        # Set button width and height relative to the screen size
        w = self.screen_width // 4
        h = self.screen_height // 5

        # Create rounded buttons using grid
        self.create_rounded_button_grid("Check sets", self.Nakijken_OK, w, h, container=button_container, row=0, column=0, Y=20, X=0)
        self.create_rounded_button_grid("Overview OQ", self.Overzicht_OK, w, h, container=button_container, row=1, column=0, Y=20, X=0)

        # Allow the buttons to expand and fill space
        button_container.grid_rowconfigure(0, weight=1)  # Allow first button to expand
        button_container.grid_rowconfigure(1, weight=1)  # Allow second button to expand
        button_container.grid_columnconfigure(0, weight=1)  # Center the buttons horizontally


    def create_rounded_button(self, text, command, w, h, container, Y=0, X=0, Side=None, Anchor=None):
        """Helper function to create a rounded button with padding, visible edge, and custom styling."""
        button = CTkButton(container, text=text, font=self.button_font, command=command, width=w, height=h, corner_radius=20, fg_color="white", text_color="black", hover_color="#e6e6e6", border_width=2, border_color="black")
        button.pack(pady=Y, padx=X, side=Side, anchor=Anchor)
        return button
        
    def create_rounded_button_grid(self, text, command, width, height, container=None, Y=0, X=0, row=None, column=None):
        """Creates a custom rounded button and places it using grid."""
        # Use CTkButton for rounded buttons
        button = CTkButton(container, text=text, font=self.button_font, command=command, width=width, height=height, corner_radius=20, fg_color="white", text_color="black", hover_color="#e6e6e6", border_width=2, border_color="black")

        # Place the button in the grid, centered, without expanding to fill the entire width
        button.grid(row=row, column=column, padx=X, pady=Y, sticky="")  # No 'nsew', just center the button
        return button

        
        
    def clear_screen(self):
        """Remove all widgets from the main_frame."""
        for widget in self.main_frame.winfo_children():
            widget.grid_forget()  # Forget widgets placed by grid
            widget.pack_forget()  # Hide all widgets in main_frame
            widget.destroy()  # Destroy the widgets to fully remove them
        
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None

        # Clear camera view and reset
        #self.label.config(image="")
        #self.label.image = None
        
        # Clear the status label when closing the camera
        #if hasattr(self, 'status_label') and self.status_label.winfo_exists():
        #    self.status_label.config(text="")  # Clear the status label
        #    self.label.image = None
        
        # Clear camera view and reset the status label
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text="")
                self.status_label.pack_forget()

        # Reset grid configuration to avoid leftover weight or sticky settings
        for i in range(self.main_frame.grid_size()[0]):  # Reset all columns
                self.main_frame.grid_columnconfigure(i, weight=0)
        for j in range(self.main_frame.grid_size()[1]):  # Reset all rows
                self.main_frame.grid_rowconfigure(j, weight=0)
    
    
    def Overzicht_OK(self):
        """Functionality for Overview OQ button."""
        self.clear_screen()
        
        # Add the home button when not on the Keuzemenu_OK screen
        self.add_home_button()
        
        # Add label for overview
        tk.Label(self.main_frame, text="No overview available", font=self.button_font, bg="white").pack(pady=20)
        
        # Add a 'Back' button to go back to the home menu
        self.create_rounded_button("Back", self.Keuzemenu_OK, w=200, h=100, container=self.main_frame, Y=20, Side = 'bottom')
        
    
    def Nakijken_OK(self):
        """Home screen for selecting sets and starting object detection."""
        self.clear_screen()

        self.reset_set_selection()

        # Add the home button when not on the Keuzemenu_OK screen
        self.add_home_button()

        # Create a frame for the scrollable area (sets buttons)
        sets_frame = tk.Frame(self.main_frame, bg='white')
        sets_frame.grid(row=0, column=0, sticky="nsew", padx=50, pady=20)

        # Configure the main frame to allow resizing
        self.main_frame.grid_rowconfigure(0, weight=1)  # Allocate space for the sets_frame
        self.main_frame.grid_rowconfigure(1, weight=0)  # For the Start Detection button
        self.main_frame.grid_columnconfigure(0, weight=1)  # Make the layout stretch horizontally

        # Ensure sets_frame expands properly
        sets_frame.grid_rowconfigure(0, weight=1)
        sets_frame.grid_columnconfigure(0, weight=1)  # Ensure full width

        # Create a canvas for scrolling
        canvas = tk.Canvas(sets_frame, bg='white')
        canvas.grid(row=0, column=0, sticky="nsew")

        # Create a scrollbar and link it to the canvas
        scrollbar = tk.Scrollbar(sets_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Ensure canvas expands with the frame
        sets_frame.grid_columnconfigure(0, weight=1)
        sets_frame.grid_rowconfigure(0, weight=1)

        # Create an inner frame inside the canvas where buttons will be placed
        button_container = tk.Frame(canvas, bg='white')
        #button_container.grid(row=0, column=0, sticky="nsew")
        canvas.create_window((20, 0), window=button_container, anchor='n')

        # Method to handle resizing of the canvas
        button_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Store references to buttons to manage appearance on selection
        self.set_buttons = {}

        # Define the number of columns (2 in this case)
        num_columns = 2
        sets = self.get_sets_with_status()  # Assume this returns a list of sets

        # Calculate button dimensions based on available screen size
        button_width = self.screen_width // 3  # Adjust the width
        button_height = self.screen_height // 8  # Adjust to 1/15 of the screen height
        
        # Populate buttons based on the sets available
        for i, set_name in enumerate(sets):
                # Create buttons and place them inside the grid for symmetrical alignment
                btn = self.create_rounded_button_grid(
                    set_name, 
                    lambda s=set_name: self.select_set(s), 
                    button_width, button_height, 
                    container=button_container, 
                    Y=10, X=40, 
                    row=i // num_columns, column=i % num_columns
                )

                # Store the button reference in a dictionary for easier access
                self.set_buttons[set_name] = btn

        # Ensure the columns stretch evenly and center align the button_container
        button_container.grid_columnconfigure(0, weight=1)
        button_container.grid_columnconfigure(1, weight=1)

        # Create a frame for the "Start Object Detection" button and place it below the buttons
        button_frame = tk.Frame(self.main_frame, bg="white")
        button_frame.grid(row=1, column=0, pady=0, sticky="")  # Center it using sticky="ew"

        # Calculate button dimensions based on available screen size
        button_width = self.screen_width // 3  # Adjust the width
        button_height = self.screen_height // 10  # Adjust to 1/15 of the screen height

        # Create the button with dynamic width/height
        self.create_rounded_button(
        "Start Object Detection", 
        self.start_detection, 
        button_width, 
        button_height, 
        container=button_frame, 
        Y=10
        )

        # Status label to show detected or not-detected items
        #self.status_label = tk.Label(self.main_frame, text="", bg="white", font=("Helvetica", 20))
        #self.status_label.grid(row=2, column=0, pady=20)

        # Ensure the status label and other elements are centered
        #self.main_frame.grid_columnconfigure(0, weight=1)




    def select_set(self, selected_set):
        """Handles the logic when a set is selected."""
        # Reset all buttons to the default color (white)
        for set_name, btn in self.set_buttons.items():
            btn.configure(fg_color="white")  # Reset to default color

        # Change the background color of the selected button to a darker grey
        selected_button = self.set_buttons.get(selected_set)
        if selected_button:
            selected_button.configure(fg_color="light grey")  # Highlight the selected button
            
            # Store the selected set name (without the status part) in self.set_selection
            self.set_selection = selected_set.split(' (')[0]  # Only store the set name part

        # Optionally, store the selected set in a variable for further use
        self.selected_set = selected_set


    
    
    
    
    
    
    #######################################################################
    ####### Object detection functions
    #######################################################################
    
    
    # reste the selected set to none
    def reset_set_selection(self):
        self.set_selection = "<select set>"
    
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
        selected_set = self.set_selection
        if selected_set == "<select set>":
            messagebox.showwarning("Select Set", "Please select a set to detect.")
            return
            
        # If not resuming from pause, add Close and Pause buttons
        if not self.resuming:
            self.clear_screen()

            label_frame_width=170
            label_frame_height=200
            
            # Configure grid layout to ensure expansion
            self.main_frame.grid_columnconfigure(0, weight=4)  # Camera frame takes more space
            self.main_frame.grid_columnconfigure(1, weight=1)  # Labels frame takes less space
            self.main_frame.grid_rowconfigure(0, weight=1)     # Camera and labels take up more space
            self.main_frame.grid_rowconfigure(1, weight=1)     # Buttons take up less space
            
            # Calculate button dimensions based on available screen size
            frame_width = self.screen_width // 3  # Adjust the width
            frame_height = self.screen_height // 3  # Adjust the screen height
            
            # Create a frame for the camera feed on the left
            self.camera_frame = tk.Frame(self.main_frame, bg="black", width=frame_width, height=frame_height)
            self.camera_frame.grid(row=0, column=0, rowspan=4, padx=10, pady=5, sticky="nsew")            
        
            # Initialize and start the camera if not already started
            if self.camera is None:
                self.camera = Picamera2()
                # Use the camera's full resolution (e.g., 1920x1080)
                full_resolution = (self.camera_width, self.camera_height)  # Adjust according to your camera specs
                self.camera.configure(self.camera.create_preview_configuration(main={"format": "RGB888", "size": full_resolution}, controls={"FrameRate": 15}))
                self.camera.start()

            # Create camera feed label and place it in the grid on the left side
            self.camera_label = tk.Label(self.camera_frame, bg="black")  # Assuming the camera feed is displayed here
            self.camera_label.grid(row=0, column=0, sticky="nsew")

            # Make sure the camera label expands within the camera_frame
            self.camera_frame.grid_rowconfigure(0, weight=1)
            self.camera_frame.grid_columnconfigure(0, weight=1)

            # Bind to the camera frame resize event to update the feed dynamically
            #self.camera_frame.bind("<Configure>", self.update_camera_feed)   
            #self.update_camera_feed()
            # Create a frame for the labels and make it scrollable
            self.label_frame = tk.Frame(self.main_frame, bg='white', width=label_frame_width, height=label_frame_height)
            self.label_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")

            # Add a canvas to enable scrolling
            self.canvas = tk.Canvas(self.label_frame, bg="white", width=label_frame_width, height=label_frame_height)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            self.scrollbar = tk.Scrollbar(self.label_frame, orient="vertical", command=self.canvas.yview)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.scrollable_frame = tk.Frame(self.canvas, bg="white", width=label_frame_width, height=label_frame_height)
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            )

            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)

            # Dynamically add status labels in the scrollable frame
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text="Detecting items...")
            else:
                # Create the status label if it doesn't exist
                self.status_label = tk.Label(self.scrollable_frame, text="Detecting items...", bg="white", font=("Helvetica", 20))
            
            self.status_label.pack(pady=10)  # Pack status label inside the scrollable frame
            
            # Create a frame for the buttons to arrange them horizontally
            self.button_frame = tk.Frame(self.main_frame, bg='white')
            self.button_frame.grid(row=1, column=1, columnspan=3, pady=5, sticky="ew")

            button_width = self.screen_width // 5  # Adjust this ratio for button size
            button_height = self.screen_height // 12
            
            # Create buttons and use relative placement
            self.create_rounded_button("Close Camera", self.close_camera, button_width, button_height, container=self.button_frame, Side=tk.TOP, X=5, Y=5)
            self.create_rounded_button("Save Image", self.toggle_pause, button_width, button_height, container=self.button_frame, Side=tk.TOP, X=5, Y=5)
            self.create_rounded_button("Mark Incomplete", self.mark_incomplete, button_width, button_height, container=self.button_frame, Side=tk.TOP, X=5, Y=5)

            # Make sure buttons expand vertically
            self.button_frame.grid_rowconfigure(0, weight=1)
            self.button_frame.grid_rowconfigure(1, weight=1)
            self.button_frame.grid_columnconfigure(0, weight=1) 
            
        else:
            self.resuming = False

        set_name = selected_set
        try:
            self.target_items = set(self.sets[set_name]["items"])
        except KeyError:
            messagebox.showerror("Error", f"Set '{set_name}' not found.")
            return

        self.detected_items.clear()

        self.detect_objects()  # Start detecting objects
        

    # Display the preview image scaled to the camera_frame size
    def update_camera_feed(self):
        # Capture the full-resolution image from the camera
        image = self.camera.capture_array()

        # Resize the image to fit the current camera frame size
        frame_width = self.camera_frame.winfo_width()
        frame_height = self.camera_frame.winfo_height()
        image_resized = cv2.resize(image, (frame_width, frame_height))

        # Convert image to PhotoImage and display in the camera_label
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        self.camera_label.config(image=image_tk)
        self.camera_label.image = image_tk  # Keep a reference to prevent garbage collection

    
    # Object detection process
    def detect_objects(self):
        if self.camera:
            if not self.paused:  # Check if the camera is not paused
                image = self.camera.capture_array()  # Capture image from camera
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PIL

                # Resize image to match the model input size
                image_resized = Image.fromarray(image_rgb).resize(
                    (self.input_details[0]['shape'][2], self.input_details[0]['shape'][1])
                )
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
                            
                            # Scale coordinates to current frame size
                            x1, y1, x2, y2 = int(xmin * self.camera_width), int(ymin * self.camera_height), int(xmax * self.camera_width), int(ymax * self.camera_height)
                            
                            # Draw rectangle and put text with scaled coordinates
                            image_rgb = cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            image_rgb = cv2.putText(
                                image_rgb, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
                            )

                # Resize the image to fit the current camera frame size
                frame_width = self.camera_frame.winfo_width()
                frame_height = self.camera_frame.winfo_height()
                image_resized = cv2.resize(image_rgb, (frame_width, frame_height))
            
                # Convert the processed image back to a format suitable for Tkinter
                image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
                
                # Recreate the label if it doesn't exist
                #if not hasattr(self, 'label') or not self.label.winfo_exists():
                #    self.label = tk.Label(self.main_frame)
                #    self.label.pack()  # Pack the label appropriately
                
                # Update the label with the image
                self.camera_label.config(image=image_tk)
                self.camera_label.image = image_tk  # Keep a reference to avoid garbage collection

                self.update_status()  # Update detection status on the screen

                # Continue detecting after 10ms
                self.window.after(10, self.detect_objects)



    # Save the current camera frame as an image
    def save_image_old(self):
        if not os.path.exists('Code/Saved Images'):  # Create directory if not exists
            os.makedirs('Code/Saved Images')

        # Capture current frame
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for saving

        # Launch the on-screen keyboard before showing the dialog
        os.system("wvkbd-mobintl &")
    
        # Create a custom dialog for the image name
        #dialog = self.CustomSaveDialog(self.window)
        #self.window.wait_window(dialog)  # Wait until the dialog is closed
        
        # Initialize the PyQt application
        app = QApplication([])
        
        # Create a custom PyQt dialog for the image name
        dialog = self.SaveImageDialog()
        if dialog.exec_() == QDialog.Accepted:
            file_name = dialog.get_filename()
        else:
            # User pressed cancel
            os.system("pkill wvkbd-mobintl")
            messagebox.showwarning("No Name Entered", "No name entered. Image not saved.")
            return

        # Close the on-screen keyboard after input is done
        os.system("pkill wvkbd-mobintl")
    
        # Prompt user for image name
        file_name = dialog.result
        if not file_name:
            cancel = messagebox.showwarning("No Name Entered", "No name entered. Image not saved.")
            return

        if not file_name.endswith('.png'):  # Ensure the file name ends with .png
            file_name += '.png'

        image_path = os.path.join('Code/Saved Images', file_name)

        # Confirm overwrite if file already exists
        if os.path.exists(image_path):
            overwrite = messagebox.askyesno("Overwrite?", f"The file '{file_name}' already exists. Do you want to overwrite it?")
            if not overwrite:
                messagebox.showinfo("Save Cancelled", "Image not saved.")
                return

        # Save image to file
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Image Saved", f"Image saved to {image_path}")

    # Save the current camera frame as an image
    def save_image(self):
        # Hide the labels and buttons, and prepare input for the image name
        self.clear_right_panel()  # Function to hide the current labels and buttons

        button_width = self.screen_width // 6  # Adjust this ratio for button size
        button_height = self.screen_height // 12
            
        #self.window.attributes('-fullscreen', False)
        #self.window.attributes('-zoomed', True)
            
        # Label for the image name entry
        self.entry_label = tk.Label(self.main_frame, text="Enter a name for the image:", width=button_width // 5, font=("Helvetica", 10), bg="white")
        self.entry_label.grid(row=0, column=1, padx=10, pady=10)
            
        # Create input field and OK/CANCEL buttons
        self.image_name_entry = tk.Entry(self.main_frame, width=button_width // 6, font=("Helvetica", 10))
        self.image_name_entry.grid(row=1, column=1, padx=10, pady=10)

        # Bind focus events to show/hide the virtual keyboard
        self.image_name_entry.bind("<FocusIn>", self.show_virtual_keyboard)  # When entry gains focus
        self.image_name_entry.bind("<FocusOut>", self.hide_virtual_keyboard)  # When entry loses focus


        # Create OK and CANCEL buttons
        self.ok_button = CTkButton(self.main_frame, text="OK", font=("Helvetica", 16), command=self.save_image_to_file, width=button_width, height=button_height, corner_radius=20, fg_color="white", text_color="black", hover_color="#e6e6e6", border_width=2, border_color="black")
        self.ok_button.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.cancel_button = CTkButton(self.main_frame, text="Cancel", font=("Helvetica", 16), command=self.cancel_save_image, width=button_width, height=button_height, corner_radius=20, fg_color="white", text_color="black", hover_color="#e6e6e6", border_width=2, border_color="black")
        self.cancel_button.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

    # Function to show the virtual keyboard when entry gains focus
    def show_virtual_keyboard(self, event=None):
        #os.system("onboard &")
        #os.system("wvkbd-mobintl &")  # Launch the virtual keyboard
        sleep(0.5)  # Give some time for the keyboard to open
        #os.system("wmctrl -a onboard")  # Bring the virtual keyboard to the front
        #subprocess.Popen(['onboard'])
        p = subprocess.Popen(["onboard&"], shell=True, stdout=subprocess.PIPE, stderr= subprocess.PIPE, universal_newlines=True)
        if not "" == p.stderr.readline():
                subprocess.Popen(["onboard &"], shell=True)
        

    # Function to hide the virtual keyboard when entry loses focus
    def hide_virtual_keyboard(self, event=None):
        os.system("pkill wvkbd-mobintl")  # Close the virtual keyboard

    # Function to confirm saving the image
    def save_image_to_file(self):
        file_name = self.image_name_entry.get()

        # Validate file name
        if not file_name:
            messagebox.showwarning("No Name Entered", "Please enter a name for the image.")
            return

        # Ensure the file name ends with .png
        if not file_name.endswith('.png'):
            file_name += '.png'

        image_path = os.path.join('Code/Saved Images', file_name)

        # Confirm overwrite if file already exists
        if os.path.exists(image_path):
            overwrite = messagebox.askyesno("Overwrite?", f"The file '{file_name}' already exists. Do you want to overwrite it?")
            if not overwrite:
                messagebox.showinfo("Save Cancelled", "Image not saved.")
                return

        # Capture current frame and save the image
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        messagebox.showinfo("Image Saved", f"Image saved to {image_path}")

        # Restore the previous state
        self.restore_right_panel()

    # Function to cancel saving and return to the camera screen
    def cancel_save_image(self):
        # Restore the previous state
        self.restore_right_panel()
        
    # Function to hide the right panel content (labels and buttons)
    def clear_right_panel(self):
        # Hide the labels and buttons
        if hasattr(self, 'label_frame'):
            self.label_frame.grid_forget()
        if hasattr(self, 'button_frame'):
            self.button_frame.grid_forget()

    # Function to restore the right panel content
    def restore_right_panel(self):
                    
        # Restore the labels and buttons on the right
        self.label_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.button_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Remove the input field and buttons
        self.entry_label.grid_forget()
        self.image_name_entry.grid_forget()
        self.ok_button.grid_forget()
        self.cancel_button.grid_forget()
        
        #self.window.attributes('-fullscreen', True)

        
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

        # Create or update the status label
        if not hasattr(self, 'status_label') or not self.status_label.winfo_exists():
                # Create a custom font for the label
                custom_font = tkFont.Font(family="Helvetica", size=20, weight="bold")  # Explicitly larger font

                # Create the label with the custom font
                self.status_label = tk.Label(self.main_frame, text=status_text, bg="white", font=custom_font)
                self.status_label.pack(pady=20)  # Use pack geometry manager
        else:
                # Update the text and font of the existing label
                self.status_label.config(text=status_text)
                self.status_label.config(font=("Helvetica", 10, "bold"))  # Update font directly
                self.status_label.update_idletasks()  # Force the UI to refresh

        # Force layout update and ensure the label resizes
        self.status_label.pack_propagate(False)  # Ensure the label doesn't shrink
        self.status_label.update()  # Force the UI to refresh with new settings

        # If all items are detected, mark set as complete
        if all_detected:
            messagebox.showinfo("Set Complete", "All target items detected!")
            # Fix: handling string instead of Tkinter variable
            selected_set = self.set_selection.split(' (')[0]
            self.sets[selected_set]["status"] = "complete"
            self.save_sets()  # Save updated set status
            self.close_camera()  # Close the camera when done

    # Mark the current set as incomplete
    def mark_incomplete(self):
        selected_set = self.set_selection.split(' (')[0]  # Fix: handle self.set_selection as a string
        self.sets[selected_set]["status"] = "incomplete"
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

        # Clear camera view and reset
        #self.label.config(image="")
        #self.label.image = None
        
        # Clear the status label when closing the camera
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            self.status_label.config(text="")  # Clear the status label
            self.label.image = None
    

        # Go back to the Nakijken_OK screen or main menu
        self.Nakijken_OK()
        
        
    # Pause and resume detection by saving a snapshot and resuming
    def toggle_pause(self):
        self.display_raw_image()  # Display raw image while pausing
        self.paused = True    
        self.save_image()  # Save image during pause
        self.clear_image_label()  # Clear the displayed raw image after save
        self.paused = False
        self.resuming = True
        self.start_detection()  # Resume detection
        

    # Clear the image label
    def clear_image_label(self):
        self.label.config(image='')  # Remove image from label
        self.label.image = None
    
    # Display raw image without bounding boxes (used when pausing)
    def display_raw_image(self):
        image = self.camera.capture_array()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize the image to fit the current camera frame size
        frame_width = self.camera_frame.winfo_width()
        frame_height = self.camera_frame.winfo_height()
        image_resized = cv2.resize(image_rgb, (frame_width, frame_height))
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        
        # Update the label with the image
        self.camera_label.config(image=image_tk)
        self.camera_label.image = image_tk  # Keep a reference to avoid garbage collection


    # Reset the UI to initial state
    def reset_to_start_screen(self):
        # Reset camera feed and buttons to start screen
        self.label.config(image="")
        self.label.image = None
        self.status_label.config(text="")
        self.start_button.pack(pady=20)

        if hasattr(self, "close_button"):
            self.close_button.pack_forget()
        if hasattr(self, "incomplete_button"):
            self.incomplete_button.pack_forget()
        if hasattr(self, "pause_button"):
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
        #print("Entering sleep mode...")
        self.sleeping = True
        
        sleep(1)

        # Turn off the screen (depends on display configuration) FOR WAYLAND
        #os.system("wlr-randr --output DSI-1 --off")
        
        # Turn the screen off using xset FOR X11
        os.system("xset dpms force off")  # Use xset to turn the display off

        # Start mouse and keyboard listeners to detect wake-up event
        mouse_listener = mouse.Listener(on_click=self.wake_up)
        keyboard_listener = keyboard.Listener(on_press=self.wake_up)

        mouse_listener.start()
        keyboard_listener.start()
        
        # Start listening for touch events
        self.listen_for_touch_events()

    def listen_for_touch_events(self):
        """Listen for touch events using libinput debug-events."""
        process = subprocess.Popen(['sudo', 'libinput', 'debug-events'], stdout=subprocess.PIPE)
        while self.sleeping:
            line = process.stdout.readline().decode('utf-8')
            if 'TOUCH_DOWN' in line:
                self.wake_up()
                process.terminate()  # Stop the process after waking up

    # Wake up from sleep mode: re-enable display and stop listeners
    def wake_up(self, *args):
        #print("Waking up from sleep mode...")
        self.sleeping = False

        # Turn the screen back on FOR WAYLAND
        #os.system("wlr-randr --output DSI-1 --on")
        
        # Turn the screen back on using xset FOR X11
        os.system("xset dpms force on")  # Use xset to turn the display back on

        # Stop mouse and keyboard listeners
        if self.mouse_listener is not None:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

        return False  # Stop listeners
        
    

# Main function: Initialize Tkinter window and run the CameraApp
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()  # Start the GUI event loop
