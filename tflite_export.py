from ultralytics import YOLO
import torch

# # Load YOLO11s model
model = YOLO("yolo11s.pt")  # Pytorch pre-trained model

# # Export the model to tflite with the correct input size and dynamic batch
model.export(format="tflite", int8=True, imgsz=320)

print("YOLO Model Successfully Exported to TFLite format")