# a virtual environment (venv) needs to be created on the Pi 5 to install the pynput library
# use command: python3 -m venv <venv name> --system-site-packages
# the last part of the command copies all of the existing packages into the venv, so we do not need to reinstall everything
#activate using: source /<pathtovenv>/bin/activate
# the venv needs to be activated to start this script

from picamera2 import Picamera2, Preview
import os
import tkinter as tk
import threading
from time import sleep
from pynput import mouse, keyboard
import subprocess

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")
        
        #full screen
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        self.picam2 = Picamera2()
        
        #camera button frame
        self.frame = tk.Frame(root)
        #self.frame.pack(expand=True)
        self.frame.grid(row=0, column=0, sticky="nsew")
        
        self.camera_button = tk.Button(self.frame, text="Open Camera", command=self.open_camera, width=20, height=2)
        self.camera_button.grid(row=0, column=0, padx=20, pady=20)
        #self.button = tk.Button(root, text="Open Camera", command=self.open_camera, width=20, height=2)
        #self.button.pack(expand=True)
        
        self.sleep_button = tk.Button(self.frame, text="Sleep", command=self.sleep_mode, width=20, height=2)
        self.sleep_button.grid(row=1, column=0, padx=20, pady=20)
        
        # configure grid to center the buttons
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.mouse_listener = None
        self.keyboard_listener = None
        self.sleeping = False
    
    def open_camera(self):
        try:
            print("starting preview")
            self.picam2.start_preview(Preview.QTGL)
            self.picam2.start()
            sleep(10)
            self.picam2.stop_preview()
            self.picam2.stop()
            print("preview stopped")
        except Exception as e:
            print(f"Error: {e}")
            
    def on_closing(self):
        print("Closing camera")
        self.picam2.close()
        self.root.destroy()
    
    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)
        
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
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

