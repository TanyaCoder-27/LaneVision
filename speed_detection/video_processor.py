

# import cv2
# import numpy as np
# import easyocr
# from ultralytics import YOLO
# import pandas as pd
# import os
# from django.conf import settings
# from collections import defaultdict
# import time
# import threading
# from queue import Queue


# class VehicleSpeedDetector:
#     def __init__(self, progress_callback=None):
#         # Load YOLO model for vehicle detection with optimizations
#         self.model = YOLO('yolov8n.pt')
#         self.model.fuse()  # Fuse model for faster inference
        
#         # Initialize OCR reader only once and use threading
#         self.ocr_reader = easyocr.Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
#         self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
#         self.speed_limit = 90  # km/h (same as old code)
        
#         # Class names mapping for COCO dataset (used by YOLOv8)
#         self.class_names = {
#             2: 'car',
#             3: 'motorcycle', 
#             5: 'bus',
#             7: 'truck'
#         }
        
#         # Vehicle tracking
#         self.vehicle_tracks = defaultdict(list)  # Store positions over time
#         self.vehicle_speeds = {}
#         self.next_vehicle_id = 1
#         self.progress_callback = progress_callback
        
#         # Vehicle tracking improvements
#         self.inactive_tracks = {}  # Store tracks that are no longer active
#         self.track_last_seen = {}  # Track when a vehicle was last seen
#         self.track_max_age = 30    # Maximum number of frames to keep inactive tracks
        
#         # Speed detection parameters (from old code)
#         self.entry_times = {}  # Track when vehicles enter the detection zone
#         self.meters_between_lines = 8  # Distance between detection lines (same as old code)
        
#         # Speed detection zone (will be initialized in process_video)
#         self.speed_detection_zone = None
#         self.roc_line_1 = None  # First detection line
#         self.roc_line_2 = None  # Second detection line
        
#         # Performance optimizations
#         self.ocr_queue = Queue(maxsize=10)  # Queue for OCR processing
#         self.frame_skip = 2  # Process every 2nd frame for YOLO (maintains accuracy)
#         self.ocr_frame_skip = 15  # Run OCR every 15 frames instead of 5-10
#         self.last_detections = {}  # Cache last frame detections
        
#         # License plate cache to avoid redundant OCR
#         self.license_plate_cache = {}  # vehicle_id -> (license_plate, confidence, frame_count)
#         self.cache_validity = 60  # Cache valid for 60 frames
        
#     def get_class_name(self, class_id):
#         """Convert class ID to human-readable name"""
#         return self.class_names.get(class_id, 'unknown')
    
#     def detect_license_plate_optimized(self, vehicle_crop, vehicle_id, frame_count):
#         """Optimized license plate detection with EasyOCR and unknown fallback"""
#         try:
#             # Check cache first
#             if vehicle_id in self.license_plate_cache:
#                 cached_plate, cached_conf, cached_frame = self.license_plate_cache[vehicle_id]
#                 if frame_count - cached_frame < self.cache_validity:
#                     return cached_plate, None  # Return cached result
            
#             # Check if crop is valid
#             if vehicle_crop is None or vehicle_crop.size == 0 or vehicle_crop.shape[0] == 0 or vehicle_crop.shape[1] == 0:
#                 # Use "unknown" for invalid crops
#                 self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#                 return "unknown", None
                
#             # Enhance the image for better OCR
#             h, w = vehicle_crop.shape[:2]
#             # Always resize to improve OCR performance
#             scale_factor = max(150 / h, 150 / w)  # Increased scale factor for better OCR
#             vehicle_crop = cv2.resize(vehicle_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            
#             # Enhanced preprocessing for license plate detection
#             gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            
#             # Apply multiple enhancement steps for better OCR results
#             enhanced = cv2.bilateralFilter(gray, 11, 90, 90)
#             enhanced = cv2.equalizeHist(enhanced)
            
#             # Additional contrast enhancement
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             enhanced = clahe.apply(enhanced)
            
#             # Try to detect license plate with EasyOCR
#             import easyocr
#             if not hasattr(self, 'ocr_reader'):
#                 self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                
#             results = self.ocr_reader.readtext(enhanced, detail=1, width_ths=0.7, height_ths=0.7, 
#                                               paragraph=False, min_size=10)
            
#             license_plates = []
#             for (bbox, text, conf) in results:
#                 if conf > 0.3:  # Slightly higher threshold for better accuracy
#                     # Clean the text (remove spaces and special characters)
#                     cleaned_text = ''.join(c for c in text if c.isalnum())
#                     # Filter by typical license plate patterns
#                     if len(cleaned_text) >= 3 and any(c.isdigit() for c in cleaned_text):
#                         license_plates.append((cleaned_text, conf, bbox))
            
#             # If no license plates detected, use "unknown"
#             if not license_plates:
#                 self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#                 return "unknown", None
            
#             # Cache the result
#             license_plates.sort(key=lambda x: x[1], reverse=True)
#             best_plate = license_plates[0][0]
#             self.license_plate_cache[vehicle_id] = (best_plate, license_plates[0][1], frame_count)
#             return best_plate, license_plates[0][2]
                
#         except Exception as e:
#             print(f"License plate detection error: {e}")
#             # Use "unknown" if exception occurs
#             self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#             return "unknown", None
    
#     def calculate_distance(self, pos1, pos2):
#         """Calculate Euclidean distance between two points"""
#         return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
#     def assign_vehicle_id(self, center_x, center_y, frame_number, max_distance=50):
#         """Optimized vehicle ID assignment"""
#         current_pos = (center_x, center_y)
        
#         # Find the closest existing track
#         min_distance = float('inf')
#         assigned_id = None
        
#         # First check active tracks
#         for vehicle_id, positions in self.vehicle_tracks.items():
#             if positions:  # Check if track has positions
#                 last_pos = positions[-1]['position']
#                 distance = self.calculate_distance(current_pos, last_pos)
                
#                 if distance < max_distance and distance < min_distance:
#                     min_distance = distance
#                     assigned_id = vehicle_id
#                     self.track_last_seen[vehicle_id] = frame_number
        
#         # If no match in active tracks, check recently inactive tracks
#         if assigned_id is None:
#             for vehicle_id in list(self.inactive_tracks.keys()):
#                 last_pos = self.inactive_tracks[vehicle_id]
#                 distance = self.calculate_distance(current_pos, last_pos)
                
#                 if distance < max_distance and distance < min_distance:
#                     min_distance = distance
#                     assigned_id = vehicle_id
#                     self.track_last_seen[vehicle_id] = frame_number
#                     del self.inactive_tracks[vehicle_id]
        
#         # If still no match, create new vehicle
#         if assigned_id is None:
#             assigned_id = self.next_vehicle_id
#             self.next_vehicle_id += 1
#             self.track_last_seen[assigned_id] = frame_number
        
#         return assigned_id
    
#     def calculate_speed(self, vehicle_id, current_pos, frame_number, fps):
#         """Calculate vehicle speed using the approach from the old code"""
#         # Add current position to track for history
#         self.vehicle_tracks[vehicle_id].append({
#             'position': current_pos,
#             'frame': frame_number,
#             'timestamp': frame_number / fps
#         })
        
#         # Get current y position
#         center_x, center_y = current_pos
        
#         # Check if vehicle is already in the speed dictionary
#         if vehicle_id in self.vehicle_speeds:
#             return self.vehicle_speeds[vehicle_id]
            
#         # Late entry fallback: if already in ROC zone
#         if self.roc_line_1 < center_y < self.roc_line_2 and vehicle_id not in self.entry_times:
#             self.entry_times[vehicle_id] = frame_number
#             return 0
            
#         # Downward movement - vehicle crosses second line
#         elif center_y >= self.roc_line_2 and vehicle_id in self.entry_times and vehicle_id not in self.vehicle_speeds:
#             elapsed = frame_number - self.entry_times[vehicle_id]
#             time_sec = elapsed / fps
#             if time_sec > 0:  # Avoid division by zero
#                 speed = (self.meters_between_lines / time_sec) * 3.6  # m/s to km/h
#                 self.vehicle_speeds[vehicle_id] = speed
#                 return speed
                
#         # Upward movement - vehicle crosses first line
#         elif center_y <= self.roc_line_1 and vehicle_id in self.entry_times and vehicle_id not in self.vehicle_speeds:
#             elapsed = frame_number - self.entry_times[vehicle_id]
#             time_sec = elapsed / fps
#             if time_sec > 0:  # Avoid division by zero
#                 speed = (self.meters_between_lines / time_sec) * 3.6  # m/s to km/h
#                 self.vehicle_speeds[vehicle_id] = speed
#                 return speed
                
#         return self.vehicle_speeds.get(vehicle_id, 0)
    
#     def cleanup_inactive_tracks(self, frame_number):
#         """Optimized cleanup of inactive tracks"""
#         # Clean up old tracks less frequently
#         if frame_number % 30 != 0:  # Only cleanup every 30 frames
#             return
            
#         expired_tracks = []
#         for vehicle_id, last_seen in list(self.track_last_seen.items()):
#             if frame_number - last_seen > self.track_max_age:
#                 expired_tracks.append(vehicle_id)
        
#         for vehicle_id in expired_tracks:
#             if vehicle_id in self.vehicle_tracks and self.vehicle_tracks[vehicle_id]:
#                 self.inactive_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-1]['position']
            
#             # Clean up all references
#             self.track_last_seen.pop(vehicle_id, None)
#             self.license_plate_cache.pop(vehicle_id, None)
    
#     def process_video(self, input_path, output_path, csv_path):
#         """Optimized video processing with performance improvements"""
#         cap = cv2.VideoCapture(input_path)
        
#         # Optimize video capture
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Define speed detection zone using the same approach as the old code
#         self.roc_line_1 = int(0.4 * height)  # First detection line at 40% of frame height
#         self.roc_line_2 = int(0.6 * height)  # Second detection line at 60% of frame height
#         self.speed_detection_zone = (0, self.roc_line_1, width, self.roc_line_2)
        
#         # Optimized video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         detection_data = []
#         frame_count = 0
        
#         # Reset tracking variables for each video
#         self.entry_times = {}
#         self.vehicle_speeds = {}
#         self.license_plate_cache = {}  # Reset cache
        
#         print(f"Processing video: {total_frames} frames at {fps} FPS")
#         print(f"Speed detection zone: Lines at y={self.roc_line_1} and y={self.roc_line_2}")
#         print(f"Distance between lines: {self.meters_between_lines} meters")
#         print(f"Performance optimizations: Frame skip={self.frame_skip}")
        
#         start_time = time.time()
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             # Update progress
#             progress = (frame_count / total_frames) * 100
#             if self.progress_callback:
#                 self.progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
            
#             # Skip frames for YOLO detection to improve speed
#             run_detection = (frame_count % self.frame_skip == 0)
#             current_detections = []
            
#             if run_detection:
#                 # Run YOLO detection with optimizations
#                 results = self.model(frame, verbose=False, imgsz=640, conf=0.5, iou=0.4)
                
#                 for r in results:
#                     boxes = r.boxes
#                     if boxes is not None:
#                         for box in boxes:
#                             # Check if detection is a vehicle
#                             class_id = int(box.cls[0])
#                             if class_id in self.vehicle_classes:
#                                 # Get bounding box coordinates
#                                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                                 confidence = float(box.conf[0])
                                
#                                 if confidence > 0.5:
#                                     # Calculate center position
#                                     center_x = (x1 + x2) // 2
#                                     center_y = (y1 + y2) // 2
                                    
#                                     current_detections.append({
#                                         'bbox': (x1, y1, x2, y2),
#                                         'center': (center_x, center_y),
#                                         'confidence': confidence,
#                                         'class_id': class_id
#                                     })
                
#                 # Cache detections for skipped frames
#                 self.last_detections[frame_count] = current_detections
#             else:
#                 # Use cached detections from previous frame
#                 last_frame = max([f for f in self.last_detections.keys() if f < frame_count], default=frame_count-1)
#                 current_detections = self.last_detections.get(last_frame, [])
            
#             # Optimized cleanup
#             self.cleanup_inactive_tracks(frame_count)
            
#             # Draw speed detection zone lines (from old code)
#             cv2.line(frame, (0, self.roc_line_1), (width, self.roc_line_1), (0, 0, 255), 2)
#             cv2.line(frame, (0, self.roc_line_2), (width, self.roc_line_2), (0, 0, 255), 2)
            
#             # Annotate frame count (from old code)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(frame, f"Frame: {frame_count}", (10, 30), font, 0.5, (255, 255, 255), 1)
#             cv2.putText(frame, f"Objects: {len(current_detections)}", (10, 70), font, 0.5, (255, 255, 255), 1)
#             cv2.putText(frame, "Speed Detection Zone", (50, self.roc_line_2 + 40), font, 0.6, (0, 0, 255), 2)
            
#             # Process detections
#             for detection in current_detections:
#                 x1, y1, x2, y2 = detection['bbox']
#                 center_x, center_y = detection['center']
                
#                 # Skip near edges (from old code)
#                 if center_y < 40 or center_y > height - 40:
#                     continue
                
#                 # Assign vehicle ID
#                 vehicle_id = self.assign_vehicle_id(center_x, center_y, frame_count)
                
#                 # Calculate speed
#                 speed = self.calculate_speed(vehicle_id, (center_x, center_y), frame_count, fps)
                
#                 # Optimized license plate detection - run less frequently
#                 license_plate = None
#                 license_plate_bbox = None
                
#                 # Run OCR much less frequently and only on vehicles with calculated speed
#                 should_run_ocr = (frame_count % self.ocr_frame_skip == 0 and 
#                                 speed > 0 and 
#                                 (y2 - y1) * (x2 - x1) > 5000)  # Only on larger vehicles
                
#                 if should_run_ocr:
#                     vehicle_crop = frame[y1:y2, x1:x2]
#                     if vehicle_crop.size > 0:
#                         license_plate, license_plate_bbox = self.detect_license_plate_optimized(
#                             vehicle_crop, vehicle_id, frame_count
#                         )
#                 else:
#                     # Get from cache
#                     if vehicle_id in self.license_plate_cache:
#                         cached_plate, _, cached_frame = self.license_plate_cache[vehicle_id]
#                         if frame_count - cached_frame < self.cache_validity:
#                             license_plate = cached_plate
                
#                 # Determine if overspeed
#                 is_overspeed = speed > self.speed_limit
                
#                 # Draw bounding box (red for overspeed, green for normal)
#                 color = (0, 255, 0) if not is_overspeed else (0, 0, 255)
#                 thickness = 2 if is_overspeed else 1
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
#                 # Draw vehicle ID
#                 id_text = f"ID:{int(vehicle_id)}"
#                 cv2.putText(frame, id_text, (x1, y1-10), font, 0.4, (0, 255, 255), 1)
                
#                 # Show speed if calculated
#                 if vehicle_id in self.vehicle_speeds:
#                     spd = round(self.vehicle_speeds[vehicle_id], 1)
#                     color = (0, 255, 0) if spd <= self.speed_limit else (0, 0, 255)
#                     cv2.putText(frame, f"{spd} km/h", (x1, y2+25), font, 0.6, color, 2)
#                     if spd > self.speed_limit:
#                         cv2.putText(frame, "OVERSPEED", (x1, y2+50), font, 0.6, color, 2)
#                 else:
#                     cv2.putText(frame, "Detecting...", (x1, y2+25), font, 0.4, (200, 200, 200), 1)         
#                 # Draw license plate if detected
#                 if license_plate:
#                     # Draw license plate text with more visibility (moved to left side of bounding box)
#                     cv2.putText(frame, f"LP: {license_plate}", (x1, y1-15), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
#                     # If we have the license plate bounding box, draw it on the vehicle crop
#                     if license_plate_bbox is not None:
#                         # Convert relative bbox coordinates to absolute frame coordinates
#                         try:
#                             # License plate bbox is relative to vehicle crop
#                             lp_x1, lp_y1 = int(license_plate_bbox[0][0]), int(license_plate_bbox[0][1])
#                             lp_x2, lp_y2 = int(license_plate_bbox[2][0]), int(license_plate_bbox[2][1])
                            
#                             # Convert to absolute coordinates in the frame
#                             abs_lp_x1, abs_lp_y1 = x1 + lp_x1, y1 + lp_y1
#                             abs_lp_x2, abs_lp_y2 = x1 + lp_x2, y1 + lp_y2
                            
#                             # Draw license plate bounding box with a distinct color
#                             cv2.rectangle(frame, (abs_lp_x1, abs_lp_y1), (abs_lp_x2, abs_lp_y2), (255, 255, 0), 1)
#                         except (IndexError, TypeError) as e:
#                             # Skip drawing if there's an issue with the bounding box
#                             pass
                
#                 # Store detection data only if speed > 0
#                 if speed > 0:
#                     # Get license plate from cache if not detected in current frame
#                     if not license_plate and vehicle_id in self.license_plate_cache:
#                         license_plate = self.license_plate_cache[vehicle_id][0]
                    
#                     # Ensure we have a license plate value
#                     if not license_plate:
#                         license_plate = f"ABC{vehicle_id:03d}"  # Generate a sample plate if none exists
                    
#                     detection_record = {
#                         'frame_number': frame_count,
#                         'vehicle_id': vehicle_id,
#                         'speed': round(speed, 2),
#                         'license_plate': license_plate,  # Always use a valid license plate
#                         'license_plate_confidence': self.license_plate_cache.get(vehicle_id, (None, 0.95, 0))[1],
#                         'is_overspeed': is_overspeed,
#                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
#                         'confidence': round(detection['confidence'], 2),
#                         'vehicle_class': self.get_class_name(detection['class_id']),
#                         'timestamp': frame_count / fps
#                     }
#                     detection_data.append(detection_record)
            
#             # Draw speed detection zone
#             zone_x1, zone_y1, zone_x2, zone_y2 = self.speed_detection_zone
#             cv2.line(frame, (zone_x1, zone_y1), (zone_x2, zone_y1), (0, 0, 255), 1)
#             cv2.line(frame, (zone_x1, zone_y2), (zone_x2, zone_y2), (0, 0, 255), 1)
#             cv2.putText(frame, "Speed Detection Zone", (10, zone_y1-10), 
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
#             # Add frame info
#             info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(current_detections)}"
#             cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
#             out.write(frame)
            
#             # Print progress every 100 frames
#             if frame_count % 100 == 0:
#                 elapsed = time.time() - start_time
#                 fps_processing = frame_count / elapsed
#                 eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
#                 print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f} | ETA: {eta:.1f}s")
        
#         cap.release()
#         out.release()
        
#         # Save detection data to CSV with unique vehicle IDs
#         if detection_data:
#             df = pd.DataFrame(detection_data)
            
#             # Get unique vehicle IDs and keep only the detection with highest speed for each ID
#             unique_df = df.loc[df.groupby('vehicle_id')['speed'].idxmax()]
            
#             # Ensure license plate column exists and is properly formatted
#             if 'license_plate' in unique_df.columns:
#                 # Replace empty, N/A or "unknown" license plates with 'Unknown'
#                 unique_df['license_plate'] = unique_df['license_plate'].apply(
#                     lambda x: 'Unknown' if pd.isna(x) or x == 'N/A' or x == '' or x.lower() == 'unknown' else x
#                 )
#             else:
#                 # Add license plate column if it doesn't exist
#                 unique_df['license_plate'] = 'Unknown'
            
#             # Reorder columns to ensure license plate is visible
#             cols = unique_df.columns.tolist()
#             if 'license_plate' in cols:
#                 cols.remove('license_plate')
#                 # Insert license_plate after vehicle_id
#                 vehicle_id_index = cols.index('vehicle_id') if 'vehicle_id' in cols else 0
#                 cols.insert(vehicle_id_index + 1, 'license_plate')
#                 unique_df = unique_df[cols]
            
#             # Save the filtered data with unique vehicle IDs
#             unique_df.to_csv(csv_path, index=False)
#             print(f"Saved {len(unique_df)} unique vehicle detections to CSV")
        
#         processing_time = time.time() - start_time
#         print(f"Video processing completed in {processing_time:.2f} seconds")
        
#         # Cleanup
#         self.last_detections.clear()
#         self.license_plate_cache.clear()
        
#         return len(detection_data)


# import cv2
# import numpy as np
# import easyocr
# from ultralytics import YOLO
# import pandas as pd
# import os
# from django.conf import settings
# from collections import defaultdict
# import time
# import threading
# from queue import Queue
# import re

# class VehicleSpeedDetector:
#     def __init__(self, progress_callback=None):
#         # Load YOLO model for vehicle detection with optimizations
#         self.model = YOLO('yolov8n.pt')
#         self.model.fuse()  # Fuse model for faster inference
        
#         # Initialize OCR reader with optimized settings for license plates
#         self.ocr_reader = easyocr.Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
#         self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
#         self.speed_limit = 90  # km/h (same as old code)
        
#         # Class names mapping for COCO dataset (used by YOLOv8)
#         self.class_names = {
#             2: 'car',
#             3: 'motorcycle', 
#             5: 'bus',
#             7: 'truck'
#         }
        
#         # Vehicle tracking
#         self.vehicle_tracks = defaultdict(list)  # Store positions over time
#         self.vehicle_speeds = {}
#         self.next_vehicle_id = 1
#         self.progress_callback = progress_callback
        
#         # Vehicle tracking improvements
#         self.inactive_tracks = {}  # Store tracks that are no longer active
#         self.track_last_seen = {}  # Track when a vehicle was last seen
#         self.track_max_age = 30    # Maximum number of frames to keep inactive tracks
        
#         # Speed detection parameters (from old code)
#         self.entry_times = {}  # Track when vehicles enter the detection zone
#         self.meters_between_lines = 8  # Distance between detection lines (same as old code)
        
#         # Speed detection zone (will be initialized in process_video)
#         self.speed_detection_zone = None
#         self.roc_line_1 = None  # First detection line
#         self.roc_line_2 = None  # Second detection line
        
#         # Performance optimizations
#         self.ocr_queue = Queue(maxsize=10)  # Queue for OCR processing
#         self.frame_skip = 2  # Process every 2nd frame for YOLO (maintains accuracy)
#         self.ocr_frame_skip = 10  # Run OCR every 10 frames instead of 15 for better accuracy
#         self.last_detections = {}  # Cache last frame detections
        
#         # License plate cache to avoid redundant OCR
#         self.license_plate_cache = {}  # vehicle_id -> (license_plate, confidence, frame_count)
#         self.cache_validity = 45  # Cache valid for 45 frames
        
#         # License plate patterns for better recognition
#         self.license_patterns = [
#             r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # Indian format: AB12CD3456
#             r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',    # Indian format: AB12C3456
#             r'^[A-Z]{3}\d{3,4}$',            # Simple format: ABC123/ABC1234
#             r'^[A-Z]{2,3}\d{2,4}[A-Z]{0,2}$', # Flexible format
#             r'^[A-Z0-9]{4,10}$'              # General alphanumeric
#         ]
        
#     def get_class_name(self, class_id):
#         """Convert class ID to human-readable name"""
#         return self.class_names.get(class_id, 'unknown')
    
#     def preprocess_for_ocr(self, img):
#         """Enhanced image preprocessing specifically for license plate OCR"""
#         try:
#             # Convert to grayscale if needed
#             if len(img.shape) == 3:
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray = img.copy()
            
#             # Resize for better OCR (larger images work better)
#             h, w = gray.shape
#             if h < 50 or w < 150:
#                 scale_x = max(2.0, 150 / w)
#                 scale_y = max(2.0, 50 / h)
#                 scale = min(scale_x, scale_y)
#                 gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
#             # Noise reduction
#             denoised = cv2.bilateralFilter(gray, 11, 17, 17)
            
#             # Contrast enhancement
#             clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#             enhanced = clahe.apply(denoised)
            
#             # Sharpening kernel
#             kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#             sharpened = cv2.filter2D(enhanced, -1, kernel)
            
#             # Gaussian blur to smooth
#             blurred = cv2.GaussianBlur(sharpened, (1, 1), 0)
            
#             return blurred
            
#         except Exception as e:
#             print(f"Preprocessing error: {e}")
#             return img
    
#     def is_valid_license_plate(self, text):
#         """Check if the detected text matches license plate patterns"""
#         if not text or len(text) < 4:
#             return False
            
#         # Clean the text - remove spaces and special characters except hyphens
#         cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
#         # Must have both letters and numbers
#         has_letters = bool(re.search(r'[A-Z]', cleaned))
#         has_numbers = bool(re.search(r'\d', cleaned))
        
#         if not (has_letters and has_numbers):
#             return False
        
#         # Check against patterns
#         for pattern in self.license_patterns:
#             if re.match(pattern, cleaned):
#                 return True
        
#         # Fallback: reasonable length with mix of letters and numbers
#         if 4 <= len(cleaned) <= 12 and has_letters and has_numbers:
#             return True
            
#         return False
    
#     def detect_license_plate_optimized(self, vehicle_crop, vehicle_id, frame_count):
#         """Improved license plate detection with better OCR and validation"""
#         try:
#             # Check cache first
#             if vehicle_id in self.license_plate_cache:
#                 cached_plate, cached_conf, cached_frame = self.license_plate_cache[vehicle_id]
#                 if frame_count - cached_frame < self.cache_validity and cached_plate != "unknown":
#                     return cached_plate, None
            
#             # Validate input
#             if vehicle_crop is None or vehicle_crop.size == 0:
#                 self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#                 return "unknown", None
            
#             h, w = vehicle_crop.shape[:2]
#             if h < 20 or w < 40:  # Too small for meaningful OCR
#                 self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#                 return "unknown", None
            
#             # Enhanced preprocessing
#             processed_img = self.preprocess_for_ocr(vehicle_crop)
            
#             # Try OCR with optimized settings for license plates
#             try:
#                 results = self.ocr_reader.readtext(
#                     processed_img, 
#                     detail=1,
#                     width_ths=0.5,    # Lower width threshold
#                     height_ths=0.5,   # Lower height threshold
#                     paragraph=False,
#                     min_size=8,       # Smaller minimum size
#                     text_threshold=0.4, # Lower text threshold
#                     low_text=0.3,     # Lower low text threshold
#                     link_threshold=0.3, # Lower link threshold
#                     canvas_size=2240,  # Larger canvas
#                     mag_ratio=1.5     # Higher magnification
#                 )
#             except Exception as ocr_error:
#                 print(f"OCR error for vehicle {vehicle_id}: {ocr_error}")
#                 self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#                 return "unknown", None
            
#             # Process OCR results
#             valid_plates = []
            
#             for (bbox, text, conf) in results:
#                 if conf > 0.2:  # Lower confidence threshold
#                     # Clean and validate text
#                     cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
                    
#                     if self.is_valid_license_plate(cleaned_text):
#                         # Calculate area of detection
#                         bbox_area = 0
#                         if len(bbox) >= 4:
#                             try:
#                                 x_coords = [point[0] for point in bbox]
#                                 y_coords = [point[1] for point in bbox]
#                                 width = max(x_coords) - min(x_coords)
#                                 height = max(y_coords) - min(y_coords)
#                                 bbox_area = width * height
#                             except:
#                                 bbox_area = len(cleaned_text) * 100  # Fallback
                        
#                         valid_plates.append((cleaned_text, conf, bbox, bbox_area))
            
#             # If no valid plates detected, try with different preprocessing
#             if not valid_plates:
#                 # Try with different preprocessing approach
#                 try:
#                     # Alternative preprocessing: edge detection + morphology
#                     alt_processed = cv2.adaptiveThreshold(
#                         processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                         cv2.THRESH_BINARY, 11, 2
#                     )
                    
#                     results_alt = self.ocr_reader.readtext(
#                         alt_processed, 
#                         detail=1,
#                         width_ths=0.3,
#                         height_ths=0.3,
#                         paragraph=False,
#                         min_size=5
#                     )
                    
#                     for (bbox, text, conf) in results_alt:
#                         if conf > 0.15:  # Even lower threshold for alternative method
#                             cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
#                             if self.is_valid_license_plate(cleaned_text):
#                                 valid_plates.append((cleaned_text, conf, bbox, 100))
                                
#                 except Exception as alt_error:
#                     pass  # Continue with original results
            
#             # Select best plate
#             if not valid_plates:
#                 self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#                 return "unknown", None
            
#             # Sort by confidence and area (prefer larger, more confident detections)
#             valid_plates.sort(key=lambda x: (x[1] * 0.7 + (x[3] / 10000) * 0.3), reverse=True)
#             best_plate, best_conf, best_bbox, _ = valid_plates[0]
            
#             # Cache the result
#             self.license_plate_cache[vehicle_id] = (best_plate, best_conf, frame_count)
#             return best_plate, best_bbox
                
#         except Exception as e:
#             print(f"License plate detection error for vehicle {vehicle_id}: {e}")
#             self.license_plate_cache[vehicle_id] = ("unknown", 0.0, frame_count)
#             return "unknown", None
    
#     def calculate_distance(self, pos1, pos2):
#         """Calculate Euclidean distance between two points"""
#         return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
#     def assign_vehicle_id(self, center_x, center_y, frame_number, max_distance=50):
#         """Optimized vehicle ID assignment"""
#         current_pos = (center_x, center_y)
        
#         # Find the closest existing track
#         min_distance = float('inf')
#         assigned_id = None
        
#         # First check active tracks
#         for vehicle_id, positions in self.vehicle_tracks.items():
#             if positions:  # Check if track has positions
#                 last_pos = positions[-1]['position']
#                 distance = self.calculate_distance(current_pos, last_pos)
                
#                 if distance < max_distance and distance < min_distance:
#                     min_distance = distance
#                     assigned_id = vehicle_id
#                     self.track_last_seen[vehicle_id] = frame_number
        
#         # If no match in active tracks, check recently inactive tracks
#         if assigned_id is None:
#             for vehicle_id in list(self.inactive_tracks.keys()):
#                 last_pos = self.inactive_tracks[vehicle_id]
#                 distance = self.calculate_distance(current_pos, last_pos)
                
#                 if distance < max_distance and distance < min_distance:
#                     min_distance = distance
#                     assigned_id = vehicle_id
#                     self.track_last_seen[vehicle_id] = frame_number
#                     del self.inactive_tracks[vehicle_id]
        
#         # If still no match, create new vehicle
#         if assigned_id is None:
#             assigned_id = self.next_vehicle_id
#             self.next_vehicle_id += 1
#             self.track_last_seen[assigned_id] = frame_number
        
#         return assigned_id
    
#     def calculate_speed(self, vehicle_id, current_pos, frame_number, fps):
#         """Calculate vehicle speed using the approach from the old code"""
#         # Add current position to track for history
#         self.vehicle_tracks[vehicle_id].append({
#             'position': current_pos,
#             'frame': frame_number,
#             'timestamp': frame_number / fps
#         })
        
#         # Get current y position
#         center_x, center_y = current_pos
        
#         # Check if vehicle is already in the speed dictionary
#         if vehicle_id in self.vehicle_speeds:
#             return self.vehicle_speeds[vehicle_id]
            
#         # Late entry fallback: if already in ROC zone
#         if self.roc_line_1 < center_y < self.roc_line_2 and vehicle_id not in self.entry_times:
#             self.entry_times[vehicle_id] = frame_number
#             return 0
            
#         # Downward movement - vehicle crosses second line
#         elif center_y >= self.roc_line_2 and vehicle_id in self.entry_times and vehicle_id not in self.vehicle_speeds:
#             elapsed = frame_number - self.entry_times[vehicle_id]
#             time_sec = elapsed / fps
#             if time_sec > 0:  # Avoid division by zero
#                 speed = (self.meters_between_lines / time_sec) * 3.6  # m/s to km/h
#                 self.vehicle_speeds[vehicle_id] = speed
#                 return speed
                
#         # Upward movement - vehicle crosses first line
#         elif center_y <= self.roc_line_1 and vehicle_id in self.entry_times and vehicle_id not in self.vehicle_speeds:
#             elapsed = frame_number - self.entry_times[vehicle_id]
#             time_sec = elapsed / fps
#             if time_sec > 0:  # Avoid division by zero
#                 speed = (self.meters_between_lines / time_sec) * 3.6  # m/s to km/h
#                 self.vehicle_speeds[vehicle_id] = speed
#                 return speed
                
#         return self.vehicle_speeds.get(vehicle_id, 0)
    
#     def cleanup_inactive_tracks(self, frame_number):
#         """Optimized cleanup of inactive tracks"""
#         # Clean up old tracks less frequently
#         if frame_number % 30 != 0:  # Only cleanup every 30 frames
#             return
            
#         expired_tracks = []
#         for vehicle_id, last_seen in list(self.track_last_seen.items()):
#             if frame_number - last_seen > self.track_max_age:
#                 expired_tracks.append(vehicle_id)
        
#         for vehicle_id in expired_tracks:
#             if vehicle_id in self.vehicle_tracks and self.vehicle_tracks[vehicle_id]:
#                 self.inactive_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-1]['position']
            
#             # Clean up all references
#             self.track_last_seen.pop(vehicle_id, None)
#             self.license_plate_cache.pop(vehicle_id, None)
    
#     def process_video(self, input_path, output_path, csv_path):
#         """Optimized video processing with performance improvements"""
#         cap = cv2.VideoCapture(input_path)
        
#         # Optimize video capture
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Define speed detection zone using the same approach as the old code
#         self.roc_line_1 = int(0.4 * height)  # First detection line at 40% of frame height
#         self.roc_line_2 = int(0.6 * height)  # Second detection line at 60% of frame height
#         self.speed_detection_zone = (0, self.roc_line_1, width, self.roc_line_2)
        
#         # Optimized video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         detection_data = []
#         frame_count = 0
        
#         # Reset tracking variables for each video
#         self.entry_times = {}
#         self.vehicle_speeds = {}
#         self.license_plate_cache = {}  # Reset cache
        
#         print(f"Processing video: {total_frames} frames at {fps} FPS")
#         print(f"Speed detection zone: Lines at y={self.roc_line_1} and y={self.roc_line_2}")
#         print(f"Distance between lines: {self.meters_between_lines} meters")
#         print(f"Performance optimizations: Frame skip={self.frame_skip}")
        
#         start_time = time.time()
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             # Update progress
#             progress = (frame_count / total_frames) * 100
#             if self.progress_callback:
#                 self.progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
            
#             # Skip frames for YOLO detection to improve speed
#             run_detection = (frame_count % self.frame_skip == 0)
#             current_detections = []
            
#             if run_detection:
#                 # Run YOLO detection with optimizations
#                 results = self.model(frame, verbose=False, imgsz=640, conf=0.5, iou=0.4)
                
#                 for r in results:
#                     boxes = r.boxes
#                     if boxes is not None:
#                         for box in boxes:
#                             # Check if detection is a vehicle
#                             class_id = int(box.cls[0])
#                             if class_id in self.vehicle_classes:
#                                 # Get bounding box coordinates
#                                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                                 confidence = float(box.conf[0])
                                
#                                 if confidence > 0.5:
#                                     # Calculate center position
#                                     center_x = (x1 + x2) // 2
#                                     center_y = (y1 + y2) // 2
                                    
#                                     current_detections.append({
#                                         'bbox': (x1, y1, x2, y2),
#                                         'center': (center_x, center_y),
#                                         'confidence': confidence,
#                                         'class_id': class_id
#                                     })
                
#                 # Cache detections for skipped frames
#                 self.last_detections[frame_count] = current_detections
#             else:
#                 # Use cached detections from previous frame
#                 last_frame = max([f for f in self.last_detections.keys() if f < frame_count], default=frame_count-1)
#                 current_detections = self.last_detections.get(last_frame, [])
            
#             # Optimized cleanup
#             self.cleanup_inactive_tracks(frame_count)
            
#             # Draw speed detection zone lines (from old code)
#             cv2.line(frame, (0, self.roc_line_1), (width, self.roc_line_1), (0, 0, 255), 2)
#             cv2.line(frame, (0, self.roc_line_2), (width, self.roc_line_2), (0, 0, 255), 2)
            
#             # Annotate frame count (from old code)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(frame, f"Frame: {frame_count}", (10, 30), font, 0.5, (255, 255, 255), 1)
#             cv2.putText(frame, f"Objects: {len(current_detections)}", (10, 70), font, 0.5, (255, 255, 255), 1)
#             cv2.putText(frame, "Speed Detection Zone", (50, self.roc_line_2 + 40), font, 0.6, (0, 0, 255), 2)
            
#             # Process detections
#             for detection in current_detections:
#                 x1, y1, x2, y2 = detection['bbox']
#                 center_x, center_y = detection['center']
                
#                 # Skip near edges (from old code)
#                 if center_y < 40 or center_y > height - 40:
#                     continue
                
#                 # Assign vehicle ID
#                 vehicle_id = self.assign_vehicle_id(center_x, center_y, frame_count)
                
#                 # Calculate speed
#                 speed = self.calculate_speed(vehicle_id, (center_x, center_y), frame_count, fps)
                
#                 # Optimized license plate detection - run more frequently for better results
#                 license_plate = None
#                 license_plate_bbox = None
                
#                 # Run OCR with improved frequency and conditions
#                 should_run_ocr = (frame_count % self.ocr_frame_skip == 0 and 
#                                 (y2 - y1) * (x2 - x1) > 3000)  # Lowered area threshold
                
#                 if should_run_ocr:
#                     vehicle_crop = frame[y1:y2, x1:x2]
#                     if vehicle_crop.size > 0:
#                         license_plate, license_plate_bbox = self.detect_license_plate_optimized(
#                             vehicle_crop, vehicle_id, frame_count
#                         )
#                 else:
#                     # Get from cache
#                     if vehicle_id in self.license_plate_cache:
#                         cached_plate, _, cached_frame = self.license_plate_cache[vehicle_id]
#                         if frame_count - cached_frame < self.cache_validity:
#                             license_plate = cached_plate
                
#                 # Determine if overspeed
#                 is_overspeed = speed > self.speed_limit
                
#                 # Draw bounding box (red for overspeed, green for normal)
#                 color = (0, 255, 0) if not is_overspeed else (0, 0, 255)
#                 thickness = 2 if is_overspeed else 1
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
#                 # Draw vehicle ID
#                 id_text = f"ID:{int(vehicle_id)}"
#                 cv2.putText(frame, id_text, (x1, y1-10), font, 0.4, (0, 255, 255), 1)
                
#                 # Show speed if calculated
#                 if vehicle_id in self.vehicle_speeds:
#                     spd = round(self.vehicle_speeds[vehicle_id], 1)
#                     color = (0, 255, 0) if spd <= self.speed_limit else (0, 0, 255)
#                     cv2.putText(frame, f"{spd} km/h", (x1, y2+25), font, 0.6, color, 2)
#                     if spd > self.speed_limit:
#                         cv2.putText(frame, "OVERSPEED", (x1, y2+50), font, 0.6, color, 2)
#                 else:
#                     cv2.putText(frame, "Detecting...", (x1, y2+25), font, 0.4, (200, 200, 200), 1)         
                
#                 # Draw license plate if detected
#                 if license_plate and license_plate != "unknown":
#                     # Draw license plate text with better visibility
#                     cv2.putText(frame, f"LP: {license_plate}", (x1, y1-30), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
#                     # If we have the license plate bounding box, draw it
#                     if license_plate_bbox is not None:
#                         try:
#                             # License plate bbox is relative to vehicle crop
#                             if len(license_plate_bbox) >= 4:
#                                 lp_x1 = min([pt[0] for pt in license_plate_bbox])
#                                 lp_y1 = min([pt[1] for pt in license_plate_bbox])
#                                 lp_x2 = max([pt[0] for pt in license_plate_bbox])
#                                 lp_y2 = max([pt[1] for pt in license_plate_bbox])
                                
#                                 # Convert to absolute coordinates
#                                 abs_lp_x1, abs_lp_y1 = x1 + int(lp_x1), y1 + int(lp_y1)
#                                 abs_lp_x2, abs_lp_y2 = x1 + int(lp_x2), y1 + int(lp_y2)
                                
#                                 # Draw license plate bounding box
#                                 cv2.rectangle(frame, (abs_lp_x1, abs_lp_y1), (abs_lp_x2, abs_lp_y2), (255, 255, 0), 2)
#                         except Exception as bbox_error:
#                             pass  # Skip drawing if there's an issue
                
#                 # Store detection data only if speed > 0
#                 if speed > 0:
#                     # Get license plate from cache if not detected in current frame
#                     if not license_plate or license_plate == "unknown":
#                         if vehicle_id in self.license_plate_cache:
#                             cached_plate = self.license_plate_cache[vehicle_id][0]
#                             if cached_plate != "unknown":
#                                 license_plate = cached_plate
                    
#                     # Generate fallback license plate if still unknown
#                     if not license_plate or license_plate == "unknown":
#                         license_plate = f"UNKNOWN_{vehicle_id:03d}"
                    
#                     detection_record = {
#                         'frame_number': frame_count,
#                         'vehicle_id': vehicle_id,
#                         'speed': round(speed, 2),
#                         'license_plate': license_plate,
#                         'license_plate_confidence': self.license_plate_cache.get(vehicle_id, (None, 0.0, 0))[1],
#                         'is_overspeed': is_overspeed,
#                         'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
#                         'confidence': round(detection['confidence'], 2),
#                         'vehicle_class': self.get_class_name(detection['class_id']),
#                         'timestamp': frame_count / fps
#                     }
#                     detection_data.append(detection_record)
            
#             # Draw speed detection zone
#             zone_x1, zone_y1, zone_x2, zone_y2 = self.speed_detection_zone
#             cv2.line(frame, (zone_x1, zone_y1), (zone_x2, zone_y1), (0, 0, 255), 1)
#             cv2.line(frame, (zone_x1, zone_y2), (zone_x2, zone_y2), (0, 0, 255), 1)
#             cv2.putText(frame, "Speed Detection Zone", (10, zone_y1-10), 
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
#             # Add frame info
#             info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(current_detections)}"
#             cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
#             out.write(frame)
            
#             # Print progress every 100 frames
#             if frame_count % 100 == 0:
#                 elapsed = time.time() - start_time
#                 fps_processing = frame_count / elapsed
#                 eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
#                 print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f} | ETA: {eta:.1f}s")
        
#         cap.release()
#         out.release()
        
#         # Save detection data to CSV with unique vehicle IDs
#         if detection_data:
#             df = pd.DataFrame(detection_data)
            
#             # Get unique vehicle IDs and keep only the detection with highest speed for each ID
#             unique_df = df.loc[df.groupby('vehicle_id')['speed'].idxmax()]
            
#             # Ensure license plate column exists and is properly formatted
#             if 'license_plate' in unique_df.columns:
#                 # Replace empty or N/A license plates but keep "UNKNOWN_" prefixed ones
#                 unique_df['license_plate'] = unique_df['license_plate'].apply(
#                     lambda x: 'Unknown' if pd.isna(x) or x == 'N/A' or x == '' else x
#                 )
#             else:
#                 unique_df['license_plate'] = 'Unknown'
            
#             # Reorder columns to ensure license plate is visible
#             cols = unique_df.columns.tolist()
#             if 'license_plate' in cols:
#                 cols.remove('license_plate')
#                 vehicle_id_index = cols.index('vehicle_id') if 'vehicle_id' in cols else 0
#                 cols.insert(vehicle_id_index + 1, 'license_plate')
#                 unique_df = unique_df[cols]
            
#             # Save the filtered data with unique vehicle IDs
#             unique_df.to_csv(csv_path, index=False)
#             print(f"Saved {len(unique_df)} unique vehicle detections to CSV")
        
#         processing_time = time.time() - start_time
#         print(f"Video processing completed in {processing_time:.2f} seconds")
        
#         # Cleanup
#         self.last_detections.clear()
#         self.license_plate_cache.clear()
        
#         return len(detection_data)


import cv2
import numpy as np
import pandas as pd
import os
from django.conf import settings
from collections import defaultdict
import time
import threading
import re
import ssl
import urllib3

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables to disable ultralytics online features
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLOv8_AUTODOWNLOAD'] = 'False'

from ultralytics import YOLO


class VehicleSpeedDetector:
    def __init__(self, progress_callback=None):
        # Load YOLO model for vehicle detection with optimizations
        try:
            # Ensure we're using the local model file
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt')
            if not os.path.exists(model_path):
                model_path = 'yolov8n.pt'  # Fallback to current directory
            
            self.model = YOLO(model_path)
            self.model.fuse()  # Fuse model for faster inference
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Try alternative loading method
            try:
                self.model = YOLO('yolov8n.pt')
                self.model.fuse()
            except Exception as e2:
                print(f"Failed to load YOLO model: {e2}")
                raise Exception(f"Cannot load YOLO model. Please ensure yolov8n.pt is available. Error: {e2}")
        
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.speed_limit = 90  # km/h (same as old code)
        
        # Class names mapping for COCO dataset (used by YOLOv8)
        self.class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Vehicle tracking
        self.vehicle_tracks = defaultdict(list)  # Store positions over time
        self.vehicle_speeds = {}
        self.next_vehicle_id = 1
        self.progress_callback = progress_callback
        
        # Vehicle tracking improvements
        self.inactive_tracks = {}  # Store tracks that are no longer active
        self.track_last_seen = {}  # Track when a vehicle was last seen
        self.track_max_age = 30    # Maximum number of frames to keep inactive tracks
        
        # Speed detection parameters (from old code)
        self.entry_times = {}  # Track when vehicles enter the detection zone
        self.meters_between_lines = 8  # Distance between detection lines (same as old code)
        
        # Speed detection zone (will be initialized in process_video)
        self.speed_detection_zone = None
        self.roc_line_1 = None  # First detection line
        self.roc_line_2 = None  # Second detection line
        
        # Performance optimizations
        self.frame_skip = 2  # Process every 2nd frame for YOLO (balanced)
        self.dynamic_skip = False
        self.min_skip = 2
        self.max_skip = 3
        self.last_detections = {}  # Cache last frame detections

        # Homography-based world mapping and speed smoothing
        # Try to auto-load homography.npy from project root; fall back to identity
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            candidate = os.path.join(project_root, 'homography.npy')
            if os.path.exists(candidate):
                H_loaded = np.load(candidate)
                if H_loaded.shape == (3, 3):
                    self.homography_matrix = H_loaded.astype(np.float32)
                    self._has_homography = True
                else:
                    self.homography_matrix = np.eye(3, dtype=np.float32)
                    self._has_homography = False
            else:
                self.homography_matrix = np.eye(3, dtype=np.float32)
                self._has_homography = False
        except Exception:
            self.homography_matrix = np.eye(3, dtype=np.float32)
            self._has_homography = False

        # Global scaling factor via env var SPEED_SCALE
        # Load from optional config file first, then allow env to override
        self.speed_scale = 1.0
        self.max_speed_kph = 180.0
        self.auto_calibrate = False
        # Simple pixel-to-meter scaling (higher value => more pixels per 1 meter)
        self.pixels_per_meter = 12.0
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            cfg_path = os.path.join(project_root, 'speed_config.json')
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    if 'SPEED_SCALE' in cfg:
                        self.speed_scale = float(cfg['SPEED_SCALE'])
                    if 'MAX_SPEED_KPH' in cfg:
                        self.max_speed_kph = float(cfg['MAX_SPEED_KPH'])
                    if 'AUTO_CALIBRATE' in cfg:
                        self.auto_calibrate = bool(cfg['AUTO_CALIBRATE'])
                    if 'PIXELS_PER_METER' in cfg:
                        self.pixels_per_meter = float(cfg['PIXELS_PER_METER'])
        except Exception:
            pass
        try:
            self.speed_scale = float(os.environ.get('SPEED_SCALE', str(self.speed_scale)))
        except Exception:
            pass
        try:
            ppm = os.environ.get('PIXELS_PER_METER')
            if ppm is not None:
                self.pixels_per_meter = float(ppm)
        except Exception:
            pass
        try:
            self.max_speed_kph = float(os.environ.get('MAX_SPEED_KPH', str(self.max_speed_kph)))
        except Exception:
            pass
        try:
            ac = os.environ.get('AUTO_CALIBRATE')
            if ac is not None:
                self.auto_calibrate = ac.lower() in ('1', 'true', 'yes')
        except Exception:
            pass
        self.world_tracks = defaultdict(list)  # vehicle_id -> list of dicts with world_position and frame
        self.speed_buffers = defaultdict(list)  # vehicle_id -> recent kph values for smoothing
        # Fallback per-video auto-calibration when no homography is provided
        self._meters_per_pixel = None
        self._calibration_pixels_per_sec = []  # gather early motion samples
        self._calibration_done = False
        # Auto-tuning
        self._speed_samples = []
        self._autotune_applied = False
        # Additional smoothing
        self.speed_ema = {}
        self.ema_alpha = 0.3
        # One-time display/locking
        self.vehicle_locked_speed = {}
        self.vehicle_speed_shown_once = set()

    def _save_speed_config(self):
        try:
            project_root = os.path.dirname(os.path.dirname(__file__))
            cfg_path = os.path.join(project_root, 'speed_config.json')
            import json
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump({'SPEED_SCALE': round(float(self.speed_scale), 4), 'MAX_SPEED_KPH': round(float(self.max_speed_kph), 1)}, f)
        except Exception:
            pass
    
    def adjust_speed(self, raw_speed_kph):
        """Apply global scaling factor to speed in km/h."""
        if raw_speed_kph <= 0:
            return 0.0
        return raw_speed_kph * float(self.speed_scale)
        
    def get_class_name(self, class_id):
        """Convert class ID to human-readable name"""
        return self.class_names.get(class_id, 'unknown')
    

    

    

    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def pixel_to_world(self, pt):
        """Map pixel coordinates to world coordinates using homography H."""
        if self._has_homography:
            pt_h = np.array([[pt]], dtype="float32")
            world = cv2.perspectiveTransform(pt_h, self.homography_matrix)[0][0]
            return float(world[0]), float(world[1])
        # Without homography, return pixel coordinates; distances will be scaled with meters_per_pixel
        return float(pt[0]), float(pt[1])

    def _maybe_update_auto_scale(self, pixel_distance, dt):
        """When homography is absent, estimate meters-per-pixel from early motion statistics."""
        if self._has_homography or self._calibration_done or not self.auto_calibrate:
            return
        if dt <= 0 or pixel_distance <= 0:
            return
        pixels_per_sec = float(pixel_distance) / float(dt)
        self._calibration_pixels_per_sec.append(pixels_per_sec)
        # After enough samples, compute meters-per-pixel so that median speed ~= 45 km/h
        if len(self._calibration_pixels_per_sec) >= 200:
            median_pps = np.median(self._calibration_pixels_per_sec)
            # Target median speed in m/s (45 km/h)
            target_mps = 45.0 / 3.6
            if median_pps > 0:
                mpp = target_mps / median_pps
                # Clamp to reasonable bounds to avoid extreme scales
                mpp = float(np.clip(mpp, 0.002, 0.05))
                self._meters_per_pixel = mpp
                self._calibration_done = True
    
    def assign_vehicle_id(self, center_x, center_y, frame_number, max_distance=50):
        """Optimized vehicle ID assignment"""
        current_pos = (center_x, center_y)
        
        # Find the closest existing track
        min_distance = float('inf')
        assigned_id = None
        
        # First check active tracks
        for vehicle_id, positions in self.vehicle_tracks.items():
            if positions:  # Check if track has positions
                last_pos = positions[-1]['position']
                distance = self.calculate_distance(current_pos, last_pos)
                
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    assigned_id = vehicle_id
                    self.track_last_seen[vehicle_id] = frame_number
        
        # If no match in active tracks, check recently inactive tracks
        if assigned_id is None:
            for vehicle_id in list(self.inactive_tracks.keys()):
                last_pos = self.inactive_tracks[vehicle_id]
                distance = self.calculate_distance(current_pos, last_pos)
                
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    assigned_id = vehicle_id
                    self.track_last_seen[vehicle_id] = frame_number
                    del self.inactive_tracks[vehicle_id]
        
        # If still no match, create new vehicle
        if assigned_id is None:
            assigned_id = self.next_vehicle_id
            self.next_vehicle_id += 1
            self.track_last_seen[assigned_id] = frame_number
        
        return assigned_id
    
    def calculate_speed(self, vehicle_id, pixel_bottom_center, frame_number, fps):
        """Pixel-displacement speed with moving-average smoothing and jump rejection."""
        # Keep raw pixel history for ID assignment consistency
        self.vehicle_tracks[vehicle_id].append({
            'position': pixel_bottom_center,
            'frame': frame_number,
            'timestamp': frame_number / fps
        })

        # Need at least 2 points to compute instantaneous speed
        track = self.vehicle_tracks[vehicle_id]
        if len(track) < 2:
            return 0.0
        prev = track[-2]
        curr = track[-1]
        dx = curr['position'][0] - prev['position'][0]
        dy = curr['position'][1] - prev['position'][1]
        pixel_distance = float(np.hypot(dx, dy))
        frame_diff = max(1, curr['frame'] - prev['frame'])
        dt = frame_diff / float(fps)
        if dt <= 0:
            return self.vehicle_speeds.get(vehicle_id, 0.0)

        meters = pixel_distance / max(1e-6, float(self.pixels_per_meter))
        speed_mps = meters / dt
        speed_kph = speed_mps * 3.6
        speed_kph = self.adjust_speed(speed_kph)
        # Safety cap to prevent outliers from bad calibration
        if speed_kph > self.max_speed_kph:
            speed_kph = self.max_speed_kph

        # Collect samples and auto-tune if too many near-cap speeds
        if not self._autotune_applied:
            self._speed_samples.append(speed_kph)
            if len(self._speed_samples) >= 300:
                import numpy as _np
                near_cap = [s for s in self._speed_samples if s >= 0.95 * self.max_speed_kph]
                ratio = len(near_cap) / float(len(self._speed_samples))
                med = float(_np.median(self._speed_samples))
                if ratio > 0.25 or med > 0.9 * self.max_speed_kph:
                    # Reduce scale conservatively
                    self.speed_scale = max(0.5, min(1.5, self.speed_scale * 0.85))
                    self._autotune_applied = True
                    self._save_speed_config()

        # Drop impossible jumps and tiny jitter
        if speed_kph > 200:
            return self.vehicle_speeds.get(vehicle_id, 0.0)
        if speed_kph < 5.0:
            speed_kph = 0.0

        # Moving average (5) then EMA for stable display
        rb = self.speed_buffers[vehicle_id]
        rb.append(speed_kph)
        if len(rb) > 5:
            rb.pop(0)
        ma = float(np.mean(rb)) if rb else 0.0
        prev_ema = self.speed_ema.get(vehicle_id)
        smoothed = ma if prev_ema is None else (1 - self.ema_alpha) * prev_ema + self.ema_alpha * ma
        self.speed_ema[vehicle_id] = smoothed

        # Require min track length
        if len(track) < 10:
            return 0.0

        # Lock a stable speed once (for display and CSV)
        if vehicle_id not in self.vehicle_locked_speed:
            if 5.0 <= smoothed <= self.max_speed_kph:
                self.vehicle_locked_speed[vehicle_id] = round(smoothed, 1)
        self.vehicle_speeds[vehicle_id] = smoothed
        return smoothed
    
    def cleanup_inactive_tracks(self, frame_number):
        """Optimized cleanup of inactive tracks"""
        # Clean up old tracks less frequently
        if frame_number % 30 != 0:  # Only cleanup every 30 frames
            return
            
        expired_tracks = []
        for vehicle_id, last_seen in list(self.track_last_seen.items()):
            if frame_number - last_seen > self.track_max_age:
                expired_tracks.append(vehicle_id)
        
        for vehicle_id in expired_tracks:
            if vehicle_id in self.vehicle_tracks and self.vehicle_tracks[vehicle_id]:
                self.inactive_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-1]['position']
            
            # Clean up all references
            self.track_last_seen.pop(vehicle_id, None)
    
    def process_video(self, input_path, output_path, csv_path):
        """Optimized video processing with performance improvements"""
        cap = cv2.VideoCapture(input_path)
        
        # Optimize video capture
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define speed detection zone using the same approach as the old code
        self.roc_line_1 = int(0.4 * height)  # First detection line at 40% of frame height
        self.roc_line_2 = int(0.6 * height)  # Second detection line at 60% of frame height
        self.speed_detection_zone = (0, self.roc_line_1, width, self.roc_line_2)
        
        # Optimized video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detection_data = []
        frame_count = 0
        
        # Reset tracking variables for each video
        self.entry_times = {}
        self.vehicle_speeds = {}
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print(f"Speed detection zone: Lines at y={self.roc_line_1} and y={self.roc_line_2}")
        print(f"Distance between lines: {self.meters_between_lines} meters")
        print(f"Performance optimizations: Frame skip={self.frame_skip}")
        print(f"Speed adjustment: Multiply by 2, cap at 120 km/h if > 150 km/h")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            if self.progress_callback:
                self.progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
            
            # Skip frames for YOLO detection to improve speed
            run_detection = (frame_count % self.frame_skip == 0)
            current_detections = []
            
            if run_detection:
                # Run YOLO detection with optimizations
                results = self.model(frame, verbose=False, imgsz=640, conf=0.5, iou=0.45)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Check if detection is a vehicle
                            class_id = int(box.cls[0])
                            if class_id in self.vehicle_classes:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                confidence = float(box.conf[0])
                                
                                if confidence > 0.5:
                                    # Calculate center position
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    
                                    current_detections.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'center': (center_x, center_y),
                                        'confidence': confidence,
                                        'class_id': class_id
                                    })
                
                # Cache detections for skipped frames
                self.last_detections[frame_count] = current_detections
                # Adjust frame_skip dynamically based on detected objects
                if self.dynamic_skip:
                    num = len(current_detections)
                    if num <= 3:
                        self.frame_skip = min(self.max_skip, self.frame_skip + 1)
                    elif num >= 10:
                        self.frame_skip = max(self.min_skip, self.frame_skip - 1)
            else:
                # Use cached detections from previous frame
                last_frame = max([f for f in self.last_detections.keys() if f < frame_count], default=frame_count-1)
                current_detections = self.last_detections.get(last_frame, [])
            
            # Optimized cleanup
            self.cleanup_inactive_tracks(frame_count)
            
            # Draw speed detection zone lines (from old code)
            cv2.line(frame, (0, self.roc_line_1), (width, self.roc_line_1), (0, 0, 255), 2)
            cv2.line(frame, (0, self.roc_line_2), (width, self.roc_line_2), (0, 0, 255), 2)
            
            # Annotate frame count (from old code)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Objects: {len(current_detections)}", (10, 70), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Speed Detection Zone", (50, self.roc_line_2 + 40), font, 0.6, (0, 0, 255), 2)
            
            # Process detections
            for detection in current_detections:
                x1, y1, x2, y2 = detection['bbox']
                # Use bottom-center point as ground contact point
                center_x = (x1 + x2) // 2
                center_y = y2
                
                # Skip near edges (from old code)
                if center_y < 40 or center_y > height - 40:
                    continue
                
                # Assign vehicle ID
                vehicle_id = self.assign_vehicle_id(center_x, center_y, frame_count)
                
                # Calculate homography-based smoothed speed (km/h)
                speed = self.calculate_speed(vehicle_id, (center_x, center_y), frame_count, fps)
                

                
                # Determine if overspeed
                is_overspeed = speed > self.speed_limit
                
                # Draw bounding box (red for overspeed, green for normal)
                color = (0, 255, 0) if not is_overspeed else (0, 0, 255)
                thickness = 2 if is_overspeed else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw vehicle ID (once, above box)
                id_text = f"ID:{int(vehicle_id)}"
                cv2.putText(frame, id_text, (x1, max(10, y1-12)), font, 0.5, (0, 255, 255), 1)
                
                # Determine reliability by distance to camera (bbox height proxy)
                bbox_h = max(1, y2 - y1)
                reliable = bbox_h > max(20, height * 0.05)
                
                # Determine final speed to display: lock once when reliable
                if vehicle_id in self.vehicle_locked_speed:
                    display_speed = self.vehicle_locked_speed[vehicle_id]
                elif reliable and vehicle_id in self.vehicle_speeds and self.vehicle_speeds[vehicle_id] > 0:
                    display_speed = round(self.vehicle_speeds[vehicle_id], 1)
                    self.vehicle_locked_speed[vehicle_id] = display_speed
                else:
                    display_speed = None

                # Draw speed only once after lock; otherwise, hide
                if display_speed is not None and vehicle_id not in self.vehicle_speed_shown_once:
                    color = (0, 255, 0) if display_speed <= self.speed_limit else (0, 0, 255)
                    cv2.putText(frame, f"{display_speed} km/h", (x1, y2 + 20), font, 0.6, color, 2)
                    self.vehicle_speed_shown_once.add(vehicle_id)
                

                
                # Store detection data: write locked speed if available
                final_speed = None
                if vehicle_id in self.vehicle_locked_speed:
                    final_speed = float(self.vehicle_locked_speed[vehicle_id])
                elif speed > 0:
                    final_speed = float(round(speed, 2))

                if final_speed is not None:
                    detection_record = {
                        'frame_number': frame_count,
                        'vehicle_id': vehicle_id,
                        'speed': round(final_speed, 2),
                        'is_overspeed': is_overspeed,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': round(detection['confidence'], 2),
                        'vehicle_class': self.get_class_name(detection['class_id']),
                        'timestamp': frame_count / fps
                    }
                    detection_data.append(detection_record)
            
            # Draw speed detection zone
            zone_x1, zone_y1, zone_x2, zone_y2 = self.speed_detection_zone
            cv2.line(frame, (zone_x1, zone_y1), (zone_x2, zone_y1), (0, 0, 255), 1)
            cv2.line(frame, (zone_x1, zone_y2), (zone_x2, zone_y2), (0, 0, 255), 1)
            cv2.putText(frame, "Speed Detection Zone", (10, zone_y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(current_detections)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f} | ETA: {eta:.1f}s")
        
        cap.release()
        out.release()
        
        # Save detection data to CSV with unique vehicle IDs
        if detection_data:
            df = pd.DataFrame(detection_data)
            
            # Get unique vehicle IDs and keep only the detection with highest speed for each ID
            unique_df = df.loc[df.groupby('vehicle_id')['speed'].idxmax()]
            

            
            # Save the filtered data with unique vehicle IDs
            unique_df.to_csv(csv_path, index=False)
            print(f"Saved {len(unique_df)} unique vehicle detections to CSV")
        
        processing_time = time.time() - start_time
        print(f"Video processing completed in {processing_time:.2f} seconds")
        
        # Cleanup
        self.last_detections.clear()
        
        return len(detection_data)
