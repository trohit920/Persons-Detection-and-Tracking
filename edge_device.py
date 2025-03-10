import numpy as np
import tensorflow as tf
import cv2
import time
from yolox.tracker.byte_tracker import BYTETracker

######################################
# 1. Load YOLO TFLite Model & Optimize
######################################
# Enable TFLite GPU delegation for faster inference
def load_model(tflite_model_path):
    
    try:
        # Try to use GPU acceleration
        interpreter = tf.lite.Interpreter(
            model_path=tflite_model_path,
            experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')]
        )
    except:
        # Fall back to CPU if GPU acceleration fails
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    # Allocate tensors once
    interpreter.allocate_tensors()

    # Get input & output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    print("Model Input Shape:", input_details[0]['shape'])
    return input_shape,input_details,interpreter,output_details

######################################
# 2. Tracker Configuration - Optimized
######################################
class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.4      # Slightly reduced for better detection
        self.track_buffer = 80       # Reduced to save memory
        self.match_thresh = 0.85      # Slightly reduced for faster matching
        self.mot20 = False
        self.frame_rate = 30

args = TrackerArgs()
tracker = BYTETracker(args, frame_rate=args.frame_rate)

######################################
# 3. Optimized Helper Functions
######################################
def preprocess_frame(frame, input_width, input_height):
    """Fast preprocessing with fixed size and optional downscaling."""
    # Faster resize with fixed size
    img = cv2.resize(frame, (320,320), interpolation=cv2.INTER_NEAREST)
    # Fast normalization
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

def non_max_suppression(boxes, scores, threshold=0.4):
    """Optimized NMS with early returns."""
    if len(boxes) == 0 or len(scores) == 0:
        return np.empty((0, 5))
    
    # Convert to the format expected by NMSBoxes
    boxes_array = np.array(boxes)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=threshold)
    
    if isinstance(indices, tuple) or indices is None or len(indices) == 0:
        return np.empty((0, 5))
    
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    
    # Fast array construction using list comprehension
    return np.array([[*boxes[i], scores[i]] for i in indices if i < len(boxes)], dtype=np.float32)

def postprocess_detections(output, frame_width, frame_height, conf_threshold=0.35):
    """Optimized detection extraction."""
    boxes, scores = [], []
    output_shape = output.shape
    
    # Fast path for the expected output format (most common case)
    if len(output_shape) == 3 and output_shape[1] >= 5:
        # Only process potential person detections (class 0)
        for i in range(output.shape[2]):
            conf = output[0, 4, i]
            if conf > conf_threshold:
                # Calculate coordinates once
                x, y, w, h = output[0, :4, i]
                
                # Faster box computation with fewer operations
                x1 = int((x - w/2) * frame_width)
                y1 = int((y - h/2) * frame_height)
                x2 = int((x + w/2) * frame_width)
                y2 = int((y + h/2) * frame_height)
                
                # Add valid boxes only
                if x1 < x2 and y1 < y2:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf)
    # Alternative formats - only try if necessary
    else:
        try:
            # Try the most common alternative format first
            if len(output_shape) == 4:
                predictions = output[0]
                # Process only the highest confidence predictions
                for box in predictions:
                    if box[4] > conf_threshold:
                        x, y, w, h = box[:4]
                        x1 = int((x - w/2) * frame_width)
                        y1 = int((y - h/2) * frame_height)
                        x2 = int((x + w/2) * frame_width)
                        y2 = int((y + h/2) * frame_height)
                        
                        if x1 < x2 and y1 < y2:
                            boxes.append([x1, y1, x2, y2])
                            scores.append(float(box[4]))
        except Exception as e:
            print(f"Detection processing error: {e}")
    
    return non_max_suppression(boxes, scores)

def draw_tracking_results(frame, tracked_objects, fps_display):
    """Simplified drawing function with single color."""
    # Use single green color for all boxes
    color = (0, 255, 0)
    
    # Draw each tracked object
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj.tlbr)
        track_id = obj.track_id
        
        # Draw bounding box with single color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add ID text (simplified)
        label = f"{track_id}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add FPS counter
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"People: {len(tracked_objects)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

######################################
# 4. Optimized Video Processing
######################################
# def main(vid_path,tflite_model):
#     # Extract input dimensions once
#     input_shape,input_details,interpreter,output_details = load_model(tflite_model)

#     input_width, input_height = input_shape[1], input_shape[2]
    
#     # Open video
#     video_path = vid_path
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Configure output video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
#     out = cv2.VideoWriter("optimized_tracked_output_1_org.mp4", fourcc, fps, (frame_width, frame_height))

#     # Performance tracking
#     start_time = time.time()
#     frame_count = 0
#     last_fps_update = time.time()
#     fps_display = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
    
#         infer_start = time.time()
        
#         # Preprocess with optimized function
#         image = preprocess_frame(frame, input_width, input_height)
        
#         # Run inference
#         interpreter.set_tensor(input_details[0]['index'], image)
#         interpreter.invoke()
#         output = interpreter.get_tensor(output_details[0]['index'])
        
#         # Postprocess detections
#         detections = postprocess_detections(output, frame_width, frame_height)
        
#         # Update tracker
#         if detections.shape[0] > 0:
#             tracked_objects = tracker.update(
#                 detections, 
#                 img_info=(frame_height, frame_width), 
#                 img_size=(frame_height, frame_width)
#             )
#         else:
#             # Use empty array for no detections
#             tracked_objects = tracker.update(
#                 np.empty((0, 5), dtype=np.float32),
#                 img_info=(frame_height, frame_width), 
#                 img_size=(frame_height, frame_width)
#             )
        
#         # Measure inference time
#         infer_time = time.time() - infer_start
#         if infer_time > 0:
#             instantaneous_fps = 1.0 / infer_time
#             print(f"Frame {frame_count}: {len(tracked_objects)} tracked, Inference FPS: {instantaneous_fps:.1f}")
    
#         # Draw tracking results
#         draw_tracking_results(frame, tracked_objects, instantaneous_fps)
        
#         # Write output frame
#         out.write(frame)
    
#     # Cleanup
#     cap.release()
#     out.release()
#     print(f"Processing complete. Average FPS: {frame_count / (time.time() - start_time):.2f}")
#     print(f"Processed video saved.")

# if __name__ == "__main__":
#     test_video = "1.mp4"
#     model_path = "yolo11s_saved_model/yolo11s_float32.tflite"
#     main(test_video,model_path)


import argparse

def main(vid_path, tflite_model, output_path):
    # Extract input dimensions once
    input_shape, input_details, interpreter, output_details = load_model(tflite_model)
    input_width, input_height = input_shape[1], input_shape[2]
    
    # Open video
    video_path = vid_path
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configure output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Performance tracking
    start_time = time.time()
    frame_count = 0
    last_fps_update = time.time()
    fps_display = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        infer_start = time.time()
        
        # Preprocess with optimized function
        image = preprocess_frame(frame, input_width, input_height)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Postprocess detections
        detections = postprocess_detections(output, frame_width, frame_height)
        
        # Update tracker
        if detections.shape[0] > 0:
            tracked_objects = tracker.update(
                detections,
                img_info=(frame_height, frame_width),
                img_size=(frame_height, frame_width)
            )
        else:
            # Use empty array for no detections
            tracked_objects = tracker.update(
                np.empty((0, 5), dtype=np.float32),
                img_info=(frame_height, frame_width),
                img_size=(frame_height, frame_width)
            )
            
        # Measure inference time
        infer_time = time.time() - infer_start
        if infer_time > 0:
            instantaneous_fps = 1.0 / infer_time
            print(f"Frame {frame_count}: {len(tracked_objects)} tracked, Inference FPS: {instantaneous_fps:.1f}")
            
        # Draw tracking results
        draw_tracking_results(frame, tracked_objects, instantaneous_fps)
        
        # Write output frame
        out.write(frame)
        
    # Cleanup
    cap.release()
    out.release()
    print(f"Processing complete. Average FPS: {frame_count / (time.time() - start_time):.2f}")
    print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YOLO object detection and tracking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, required=True, help='Path to TFLite model file')
    parser.add_argument('--output', type=str, required=True, help='Path for output video file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args.video, args.model, args.output)