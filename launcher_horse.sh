#!/bin/sh
# launcher.sh
# navigate to home directory, then to this directory, then execute python script, then back home

cd /
cd home/LynqoProto/Lynqo_detection
export DISPLAY=:0
sleep 10
bin/python3  Code/Lynqo_detection_14_KB_fixed.py
cd /

