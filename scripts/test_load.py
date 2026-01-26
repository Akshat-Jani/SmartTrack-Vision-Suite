# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:37:10 2025

@author: acer
"""

from ultralytics import YOLO

print("Ultralytics import OK")

try:
    m = YOLO("yolov8n.pt")
    print("Loaded model OK:", type(m), getattr(m, "model", None))
except Exception as e:
    print("Model load ERROR:", repr(e))
