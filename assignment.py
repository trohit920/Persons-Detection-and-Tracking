import cv2
import numpy as np
import time
import os
import psutil
from collections import deque, defaultdict
import argparse
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

#######################################
#        1. Configuration
#######################################

MODEL_PATH = "yolo11s.pt"      # YOLO model checkpoint
PERSON_CLASS_ID = 0            # 'person' class index in YOLO
MIN_CONFIDENCE = 0.5           # Confidence threshold for person detection
TRACK_HISTORY_SIZE = 60        # Track history for better reidentification
TRACK_COLOR = (0, 255, 0)  # Green color for all detections and tracks
MAX_PEOPLE = 10          # Upper limit for dancer ID assignment
# Store appearance features for each dancer
appearance_features = {}  # track_id -> features
appearance_history = {}   # track_id -> list of recent feature vectors
inactive_tracks = {}      # track_id -> (last_frame, features)
reidentified_tracks = {}  # current_id -> original_id
people_labels = {}        # Consistent dancer labels
# Enhanced tracker parameters
class ByteTrackArgs:
    def __init__(self):
        self.track_thresh = 0.5         # Reduced to catch more detections
        self.track_buffer = 120         # Very long buffer for dancing video
        self.match_thresh = 0.9         # High matching threshold
        self.mot20 = False
        self.frame_rate = 30

#######################################
#        2. Load Models
#######################################

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Setting up ByteTracker...")
tracker_args = ByteTrackArgs()
tracker = BYTETracker(tracker_args, frame_rate=tracker_args.frame_rate)

track_history = {}  # track_id -> deque of positions

#######################################
#    3. Enhanced Appearance Features
#######################################

def get_person_features(frame, bbox):
    """
    Extract comprehensive appearance features from a person bounding box
    Includes color histograms and grid-based features
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure coordinates are within frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract person ROI
    person_roi = frame[y1:y2, x1:x2]
    
    # 1. Get overall HSV color histogram
    hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
    hist_full = cv2.calcHist([hsv_roi], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_full, hist_full, 0, 1, cv2.NORM_MINMAX)
    
    # 2. Divide ROI into upper and lower regions for clothing separation
    height = y2 - y1
    mid_y = y1 + height // 2
    
    upper_roi = frame[y1:mid_y, x1:x2]
    if upper_roi.size > 0:
        upper_hsv = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2HSV)
        hist_upper = cv2.calcHist([upper_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist_upper, hist_upper, 0, 1, cv2.NORM_MINMAX)
    else:
        hist_upper = np.zeros((32, 32), dtype=np.float32)
        
    lower_roi = frame[mid_y:y2, x1:x2]
    if lower_roi.size > 0:
        lower_hsv = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2HSV)
        hist_lower = cv2.calcHist([lower_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist_lower, hist_lower, 0, 1, cv2.NORM_MINMAX)
    else:
        hist_lower = np.zeros((32, 32), dtype=np.float32)
    
    # 3. Shape features (aspect ratio)
    aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
    
    features = np.concatenate([
        hist_full.flatten(), 
        hist_upper.flatten(),
        hist_lower.flatten(),
        np.array([aspect_ratio])
    ])
    
    return features

#######################################
#    4. ID Management System
#######################################

def compare_features(features1, features2):
    """Compare two feature vectors and return similarity score"""
    # Use correlation coefficient for better comparison
    if features1 is None or features2 is None:
        return 0
    
    hist_len = len(features1) - 1
    corr = np.corrcoef(features1[:hist_len], features2[:hist_len])[0, 1]
    
    if np.isnan(corr):
        return 0
        
    return corr

def resolve_id_conflicts(current_tracks, current_frame):
    """
    Resolve ID conflicts and handle reidentification of PEOPLE
    who disappear and reappear in the frame
    """
    global people_labels
    
    # 1. Update inactive tracks
    active_ids = [t.track_id for t in current_tracks]
    
    # Move tracks that are no longer active to inactive
    for track_id in list(appearance_features.keys()):
        if track_id not in active_ids and track_id not in inactive_tracks:
            inactive_tracks[track_id] = (current_frame, appearance_features[track_id])
    
    # 2. Check each current track for potential reidentification
    for t in current_tracks:
        track_id = t.track_id
        
        # Skip if already processed
        if track_id in reidentified_tracks:
            continue
            
        # Skip if no appearance features
        if track_id not in appearance_features:
            continue
            
        current_features = appearance_features[track_id]
        
        # Compare with inactive tracks for reidentification
        best_match_id = None
        best_match_score = 0.85  # Threshold for considering a match
        
        for inactive_id, (last_frame, inactive_features) in inactive_tracks.items():

            if (current_frame - last_frame < 5 or 
                inactive_id in reidentified_tracks.values()):
                continue
                
            # Compare features
            similarity = compare_features(current_features, inactive_features)
            
            # If strong match found, consider it a reidentification
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = inactive_id
        
        # If match found, mark as reidentified
        if best_match_id is not None:
            reidentified_tracks[track_id] = best_match_id
            # Remove from inactive as it's now reidentified
            inactive_tracks.pop(best_match_id, None)
    
    # 3. Assign sequential dancer labels dynamically
    for t in current_tracks:
        track_id = t.track_id
        effective_id = reidentified_tracks.get(track_id, track_id)
        
        # Assign dancer label if not already assigned
        if effective_id not in people_labels:

            used_labels = set(people_labels.values())
            for i in range(1, MAX_PEOPLE + 1):
                if i not in used_labels:
                    people_labels[effective_id] = i
                    break
            
            # If we somehow exceed MAX_PEOPLE, just use the track_id
            if effective_id not in people_labels:
                people_labels[effective_id] = effective_id

def get_display_id(track_id):
    """Get the display ID for a track, considering reidentifications"""
    # First check if this ID has been reidentified
    if track_id in reidentified_tracks:
        original_id = reidentified_tracks[track_id]
    else:
        original_id = track_id
    
    # If the original ID has a dancer label, use that
    if original_id in people_labels:
        return people_labels[original_id]
    
    # Otherwise just return the original track ID
    return original_id

#######################################
#    5. Advanced Motion Analysis
#######################################

def predict_position(track_history_queue, frames_ahead=5):
    """
    Predict future position based on recent motion history
    Used to handle occlusions better
    """
    if len(track_history_queue) < 3:
        return None
    
    # Get most recent positions
    recent_positions = list(track_history_queue)[-3:]
    
    # Calculate velocity vectors
    velocities = []
    for i in range(1, len(recent_positions)):
        dx = recent_positions[i][0] - recent_positions[i-1][0]
        dy = recent_positions[i][1] - recent_positions[i-1][1]
        velocities.append((dx, dy))
    
    # Average velocity
    avg_dx = sum(v[0] for v in velocities) / len(velocities)
    avg_dy = sum(v[1] for v in velocities) / len(velocities)
    
    # Last known position
    last_x, last_y = recent_positions[-1]
    
    # Predict future position
    pred_x = int(last_x + avg_dx * frames_ahead)
    pred_y = int(last_y + avg_dy * frames_ahead)
    
    return (pred_x, pred_y)

#######################################
#    6. Performance Metrics
#######################################

class PerformanceMetrics:
    def __init__(self):
        self.time_metrics = defaultdict(list)
        self.memory_usage = []
        self.detection_count_per_frame = []
        self.track_count_per_frame = []
        self.id_switches = 0
        self.prev_track_ids = set()
        self.start_process_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        self.track_lifetimes = defaultdict(int)        # track_id -> frames alive
        self.reidentification_events = []              # List of (frame, original_id, new_id)
        self.frame_timestamps = []                     # Timestamp of each frame
        self.cpu_usage = []                            # CPU usage % per frame
        self.gpu_memory = []                           # GPU memory if available
        
        # Device info
        self.device_info = {
            "os": os.name,
            "cpu_count": os.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        }
        
        # For ID switch detection
        self.track_last_seen = {}                      # track_id -> last frame seen
        
    def start_frame(self, frame_number):
        """Record start of frame processing"""
        self.current_frame = frame_number
        self.frame_start = time.time()
        self.frame_timestamps.append(self.frame_start)
        
        # CPU usage
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        
    def end_frame(self, active_track_ids):
        """Record metrics at end of frame processing"""
        frame_end = time.time()
        self.time_metrics['total_frame'].append(frame_end - self.frame_start)
        
        # Memory usage
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        self.memory_usage.append(current_memory)
        
        # ID switch detection (more sophisticated)
        current_ids = set(active_track_ids)
        
        # Update track lifetimes
        for track_id in current_ids:
            self.track_lifetimes[track_id] += 1
            self.track_last_seen[track_id] = self.current_frame
        
        # Detect ID switches by checking disappeared/reappeared tracks
        appeared_ids = current_ids - self.prev_track_ids
        for track_id in appeared_ids:
            # Check if this is genuinely new or reappeared after short gap
            if track_id in self.track_last_seen and \
               self.current_frame - self.track_last_seen[track_id] <= 30:  # reappeared within 1s
                self.id_switches += 1
        
        self.prev_track_ids = current_ids
    
    def time_operation(self, operation_name, start_time):
        """Record time for a specific operation"""
        end_time = time.time()
        self.time_metrics[operation_name].append(end_time - start_time)
        return end_time
    
    def record_detections(self, count):
        """Record number of detections"""
        self.detection_count_per_frame.append(count)
    
    def record_tracks(self, count):
        """Record number of tracks"""
        self.track_count_per_frame.append(count)
    
    def record_reidentification(self, original_id, new_id):
        """Record a reidentification event"""
        self.reidentification_events.append((self.current_frame, original_id, new_id))
    
    def get_report(self, total_frames, duration):
        """Generate performance metrics report"""
        report = {
            "device_info": self.device_info,
            "speed": {
                "fps": {
                    "average": total_frames / sum(self.time_metrics['total_frame']),
                    "peak": 1.0 / min(self.time_metrics['total_frame']) if self.time_metrics['total_frame'] else 0,
                    "min": 1.0 / max(self.time_metrics['total_frame']) if self.time_metrics['total_frame'] else 0
                },
                "latency": {
                    "average_ms": sum(self.time_metrics['total_frame']) * 1000 / len(self.time_metrics['total_frame'])
                },
                "component_times": {}
            },
            "memory": {
                "peak_mb": max(self.memory_usage),
                "average_mb": sum(self.memory_usage) / len(self.memory_usage),
                "increase_mb": max(self.memory_usage) - self.start_process_memory
            },
            "cpu": {
                "average_percent": sum(self.cpu_usage) / len(self.cpu_usage),
                "peak_percent": max(self.cpu_usage)
            },
            "tracking": {
                "average_detections": sum(self.detection_count_per_frame) / len(self.detection_count_per_frame),
                "average_tracks": sum(self.track_count_per_frame) / len(self.track_count_per_frame),
                "id_switches": self.id_switches,
                "reidentification_events": len(self.reidentification_events),
                "average_track_lifetime": sum(self.track_lifetimes.values()) / len(self.track_lifetimes) if self.track_lifetimes else 0
            },
            "efficiency": {
                "tracks_per_second": sum(self.track_count_per_frame) / duration,
                "memory_per_track": max(self.memory_usage) / (max(max(self.track_count_per_frame), 1))
            }
        }
        
        # Add component timing breakdowns
        for component, times in self.time_metrics.items():
            if times and component != 'total_frame':
                avg_time = sum(times) / len(times) * 1000  # ms
                percentage = sum(times) / sum(self.time_metrics['total_frame']) * 100
                report["speed"]["component_times"][component] = {
                    "avg_ms": avg_time,
                    "percentage": percentage
                }
        
        return report

#######################################
#    7. Main Processing Loop
#######################################

def main():
    parser = argparse.ArgumentParser(description="Human Detection and Tracking using YOLO and ByteTrack")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video file")
    args = parser.parse_args()

    VIDEO_PATH = args.input
    OUTPUT_PATH = args.output

    print(f"Processing input video: {VIDEO_PATH}")
    print(f"Output will be saved to: {OUTPUT_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_metadata = cap.get(cv2.CAP_PROP_FPS)
    if fps_metadata < 1:
        fps_metadata = 30  # Default FPS

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Xvid codec
    out = cv2.VideoWriter(OUTPUT_PATH.replace('.mp4', '.avi'), fourcc, int(fps_metadata), (frame_width, frame_height))

    frame_count = 0
    
    # Performance metrics
    metrics = PerformanceMetrics()
    
    # Tracking state variables
    occlusion_detected = False
    occlusion_count = 0
    active_tracks_history = deque(maxlen=10)  # For tracking stability
    people_count_estimate = 0  # Will be updated dynamically
    
    # Record start time for total duration calculation
    total_start_time = time.time()

    while True:
        # Start timing for this frame
        metrics.start_frame(frame_count)
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Make a copy for visualization
        vis_frame = frame.copy()

        # --- (A) YOLO Person Detection ---
        detect_start = time.time()
        results = model(frame, imgsz=640)  # Higher resolution for better detection
        detect_end = metrics.time_operation('detection', detect_start)
        
        person_detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == PERSON_CLASS_ID and conf >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_detections.append([x1, y1, x2, y2, conf])
        
        metrics.record_detections(len(person_detections))

        # --- (B) Convert detections for ByteTrack ---
        dets = np.array(person_detections, dtype=np.float32)

        # --- (C) Run ByteTrack ---
        track_start = time.time()
        tracks = []
        active_track_ids = []
        
        if len(dets) > 0:
            img_info = (frame.shape[0], frame.shape[1])  # (height, width)
            img_size = (frame.shape[0], frame.shape[1])
            tracks = tracker.update(dets, img_info, img_size)
            active_track_ids = [t.track_id for t in tracks]
            
            # Monitor active tracks to dynamically estimate dancer count
            active_tracks_history.append(len(tracks))
            
            # Dynamically estimate the number of PEOPLE
            # Use the 75th percentile of recent track counts to be robust to occlusions
            if len(active_tracks_history) > 5:
                people_count_estimate = int(np.percentile(active_tracks_history, 75))
            else:
                people_count_estimate = max(people_count_estimate, len(tracks))
            
            # Detect potential occlusions
            avg_tracks = sum(active_tracks_history) / len(active_tracks_history)
            
            # Occlusion detection - two methods:
            # 1. If we see fewer tracks than our people estimate, potential occlusion
            if people_count_estimate > 0 and avg_tracks < people_count_estimate * 0.75:  # 25% fewer tracks than expected
                occlusion_count = min(20, occlusion_count + 1)
            # 2. If we detect sudden drop in tracks
            elif len(active_tracks_history) > 3 and avg_tracks < 0.8 * max(active_tracks_history):
                occlusion_count = min(20, occlusion_count + 1)
            else:
                occlusion_count = max(0, occlusion_count - 1)
            
            occlusion_detected = occlusion_count > 10
            
            # Record tracking metrics
            metrics.record_tracks(len(tracks))
            metrics.time_operation('tracking', track_start)
            
            # Resolve ID conflicts based on appearance
            id_resolve_start = time.time()
            resolve_id_conflicts(tracks, frame_count)
            metrics.time_operation('id_resolution', id_resolve_start)
            
            # Process each track
            vis_start = time.time()
            for t in tracks:
                track_id = t.track_id
                bbox = list(map(int, t.tlbr))  # Convert to list for easier manipulation
                
                # Extract features for this detection
                features = get_person_features(frame, bbox)
                if features is None:
                    continue
                    
                # Update appearance features
                if track_id not in appearance_features:
                    appearance_features[track_id] = features
                    appearance_history[track_id] = [features]
                    track_history[track_id] = deque(maxlen=TRACK_HISTORY_SIZE)
                else:
                    # Weighted update of features to adapt to appearance changes
                    alpha = 0.2  # Weight for new features
                    appearance_features[track_id] = (1-alpha) * appearance_features[track_id] + alpha * features
                    
                    # Store in history
                    appearance_history[track_id].append(features)
                    if len(appearance_history[track_id]) > 10:
                        appearance_history[track_id].pop(0)
                
                # Get consistent display ID
                display_id = get_display_id(track_id)
                
                # If reidentified, record the event
                if track_id in reidentified_tracks:
                    metrics.record_reidentification(reidentified_tracks[track_id], track_id)
                
                # Store center position in history
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Initialize history for the effective display ID if needed
                if display_id not in track_history:
                    track_history[display_id] = deque(maxlen=TRACK_HISTORY_SIZE)
                
                # Store position in track history using the display ID
                track_history[display_id].append((center_x, center_y))
                
                # Draw bounding box with consistent green color
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), TRACK_COLOR, 2)
                
                # Draw track ID with background for visibility
                label = f"ID {display_id}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_frame, (bbox[0], bbox[1] - 20), 
                            (bbox[0] + text_size[0], bbox[1]), TRACK_COLOR, -1)
                cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw trajectory (motion history)
                if display_id in track_history and len(track_history[display_id]) > 1:
                    # Convert history to numpy array for drawing
                    points = np.array(list(track_history[display_id]), dtype=np.int32)
                    # Draw polylines for track history
                    cv2.polylines(vis_frame, [points], False, TRACK_COLOR, 2)
                
                # If in occlusion, predict future positions
                if occlusion_detected and display_id in track_history:
                    predicted_pos = predict_position(track_history[display_id])
                    if predicted_pos:
                        # Draw predicted position as a circle
                        cv2.circle(vis_frame, predicted_pos, 5, TRACK_COLOR, -1)
                        cv2.circle(vis_frame, predicted_pos, 10, TRACK_COLOR, 2)
            
            metrics.time_operation('visualization', vis_start)

        # --- (D) Draw diagnostic information ---
        fps_val = 1 / (time.time() - metrics.frame_start + 1e-6)  # Avoid division by zero
        cv2.putText(vis_frame, f"FPS: {fps_val:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame count, detected people count and occlusion indicator
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(vis_frame, f"PEOPLE: ~{people_count_estimate}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Enhanced occlusion visualization
        if occlusion_detected:
            # Draw emphasized occlusion warning with red background
            occlusion_text = "OCCLUSION DETECTED"
            text_size = cv2.getTextSize(occlusion_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            
            # Draw a semi-transparent red background for the occlusion warning
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (10, 120 - 5), 
                        (10 + text_size[0], 120 + text_size[1] + 5), (0, 0, 200), -1)
            # Apply the overlay with transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
            
            # Draw occlusion text
            cv2.putText(vis_frame, occlusion_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add occlusion severity indicator
            severity = min(100, int(occlusion_count * 5))  # Scale to percentage
            cv2.rectangle(vis_frame, (10, 140), (10 + severity, 150), (0, 0, 255), -1)
            cv2.rectangle(vis_frame, (10, 140), (110, 150), (255, 255, 255), 1)
        
        # Memory usage and other performance metrics
        current_memory = metrics.memory_usage[-1] if metrics.memory_usage else 0
        
        # Status update
        if frame_count % 30 == 0:
            num_reidentified = len(reidentified_tracks)
            avg_detection_time = sum(metrics.time_metrics['detection'][-30:]) / min(30, len(metrics.time_metrics['detection'])) * 1000
            avg_tracking_time = sum(metrics.time_metrics['tracking'][-30:]) / min(30, len(metrics.time_metrics['tracking'])) * 1000
            print(f"Frame {frame_count} | FPS: {fps_val:.2f} | Active tracks: {len(tracks)} | Estimated PEOPLE: {people_count_estimate}")
            print(f"Performance: Det: {avg_detection_time:.1f}ms | Track: {avg_tracking_time:.1f}ms | Memory: {current_memory:.1f}MB")

        # Finalize metrics for this frame
        metrics.end_frame(active_track_ids)
        
        # --- (E) Write frame ---
        out.write(vis_frame)

    # Calculate total duration
    total_duration = time.time() - total_start_time
    
    # Generate performance report
    perf_report = metrics.get_report(frame_count, total_duration)
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print performance summary
    print("\n===== PERFORMANCE METRICS SUMMARY =====")
    print(f"✅ Processing complete. Output saved at: {OUTPUT_PATH}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average FPS: {perf_report['speed']['fps']['average']:.2f}")
    print(f"Peak FPS: {perf_report['speed']['fps']['peak']:.2f}")
    print(f"Average latency per frame: {perf_report['speed']['latency']['average_ms']:.2f} ms")
    
    print("\nComponent Times:")
    for component, data in perf_report['speed']['component_times'].items():
        print(f"  - {component}: {data['avg_ms']:.2f} ms ({data['percentage']:.1f}%)")
    
    print("\nMemory Usage:")
    print(f"  - Peak: {perf_report['memory']['peak_mb']:.2f} MB")
    print(f"  - Average: {perf_report['memory']['average_mb']:.2f} MB")
    
    print("\nTracking Performance:")
    print(f"  - Average detections per frame: {perf_report['tracking']['average_detections']:.2f}")
    print(f"  - Average tracks per frame: {perf_report['tracking']['average_tracks']:.2f}")

if __name__ == "__main__":
    main()
