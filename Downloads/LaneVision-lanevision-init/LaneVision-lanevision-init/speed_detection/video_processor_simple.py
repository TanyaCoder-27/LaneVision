import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import ssl
import urllib3

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables to disable ultralytics online features
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLOv8_AUTODOWNLOAD'] = 'False'

from ultralytics import YOLO

def process_video_simple(input_path, output_path, csv_path, progress_callback=None):
    """Simplified video processing with guaranteed browser compatibility"""
    
    # Initialize YOLO with error handling
    try:
        # Ensure we're using the local model file
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt')
        if not os.path.exists(model_path):
            model_path = 'yolov8n.pt'  # Fallback to current directory
        
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Try alternative loading method
        try:
            model = YOLO('yolov8n.pt')
        except Exception as e2:
            print(f"Failed to load YOLO model: {e2}")
            raise Exception(f"Cannot load YOLO model. Please ensure yolov8n.pt is available. Error: {e2}")
    
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Force common web-compatible dimensions
    if width > 1920:
        width = 1920
    if height > 1080:
        height = 1080
    
    # Ensure even dimensions
    width = width - (width % 2)
    height = height - (height % 2)
    
    # Try multiple codecs in order of browser compatibility
    codecs = ['mp4v', 'XVID', 'MJPG']
    out = None
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
    
    if not out or not out.isOpened():
        raise Exception("Cannot initialize video writer with any codec")
    
    detection_data = []
    frame_count = 0
    vehicle_tracks = defaultdict(list)  # pixel history per id
    world_tracks = defaultdict(list)    # world history per id
    speed_buffers = defaultdict(list)   # recent kph per id
    speed_ema = {}
    ema_alpha = 0.3
    next_id = 1

    # Load homography matrix if available; fall back to identity
    H = np.eye(3, dtype=np.float32)
    has_H = False
    try:
        project_root = os.path.dirname(os.path.dirname(__file__))
        candidate = os.path.join(project_root, 'homography.npy')
        if os.path.exists(candidate):
            H_loaded = np.load(candidate)
            if H_loaded.shape == (3, 3):
                H = H_loaded.astype(np.float32)
                has_H = True
    except Exception:
        pass

    # Optional global speed scale via env; default 1.0
    # Load from optional speed_config.json, allow env to override
    speed_scale = 1.0
    max_speed_kph = 180.0
    try:
        project_root = os.path.dirname(os.path.dirname(__file__))
        cfg_path = os.path.join(project_root, 'speed_config.json')
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                if 'SPEED_SCALE' in cfg:
                    speed_scale = float(cfg['SPEED_SCALE'])
                if 'MAX_SPEED_KPH' in cfg:
                    max_speed_kph = float(cfg['MAX_SPEED_KPH'])
    except Exception:
        pass
    try:
        speed_scale = float(os.environ.get('SPEED_SCALE', str(speed_scale)))
    except Exception:
        pass
    try:
        max_speed_kph = float(os.environ.get('MAX_SPEED_KPH', str(max_speed_kph)))
    except Exception:
        pass

    def pixel_to_world(pt):
        if has_H:
            pt_h = np.array([[pt]], dtype="float32")
            return cv2.perspectiveTransform(pt_h, H)[0][0]
        return np.array(pt, dtype=np.float32)

    # Auto-calibration when no homography: learn meters-per-pixel from motion
    meters_per_pixel = None
    calibration_pixels_per_sec = []
    calibration_done = False

    def maybe_update_auto_scale(pixel_distance, dt):
        nonlocal meters_per_pixel, calibration_done
        if has_H or calibration_done:
            return
        if dt <= 0 or pixel_distance <= 0:
            return
        pps = float(pixel_distance) / float(dt)
        calibration_pixels_per_sec.append(pps)
        if len(calibration_pixels_per_sec) >= 200:
            median_pps = float(np.median(calibration_pixels_per_sec))
            target_mps = 45.0 / 3.6  # target median 45 km/h
            if median_pps > 0:
                mpp = target_mps / median_pps
                mpp = float(np.clip(mpp, 0.002, 0.05))
                meters_per_pixel = mpp
                calibration_done = True

    def assign_vehicle_id(center_x, center_y, max_distance=50):
        current_pos = (center_x, center_y)
        min_distance = float('inf')
        assigned_id = None
        for vid, positions in vehicle_tracks.items():
            if positions:
                last_pos = positions[-1]['position']
                dist = np.hypot(current_pos[0] - last_pos[0], current_pos[1] - last_pos[1])
                if dist < max_distance and dist < min_distance:
                    min_distance = dist
                    assigned_id = vid
        nonlocal next_id
        if assigned_id is None:
            assigned_id = next_id
            next_id += 1
        return assigned_id
    
    print(f"Processing {total_frames} frames at {fps} FPS, resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Resize frame if necessary
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        # Progress update
        if progress_callback and frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
        
        # YOLO detection
        results = model(frame, verbose=False, imgsz=512)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            # Bottom-center point as ground contact
                            x_mid = (x1 + x2) // 2
                            y_max = y2
                            vehicle_id = assign_vehicle_id(x_mid, y_max)

                            # Track pixel and world positions
                            vehicle_tracks[vehicle_id].append({'position': (x_mid, y_max), 'frame': frame_count})
                            world_xy = pixel_to_world((x_mid, y_max))
                            world_tracks[vehicle_id].append({'world_position': (float(world_xy[0]), float(world_xy[1])), 'frame': frame_count})

                            # Regression-based speed over a sliding window
                            speed_kph = 0.0
                            wt = world_tracks[vehicle_id]
                            if len(wt) >= 2:
                                window = min(20, len(wt))
                                pts = wt[-window:]
                                times = np.array([p['frame'] / float(fps) for p in pts], dtype=np.float64)
                                xs = np.array([p['world_position'][0] for p in pts], dtype=np.float64)
                                ys = np.array([p['world_position'][1] for p in pts], dtype=np.float64)
                                if not has_H:
                                    prev = wt[-2]
                                    curr = wt[-1]
                                    dlast = float(np.hypot(curr['world_position'][0]-prev['world_position'][0],
                                                          curr['world_position'][1]-prev['world_position'][1]))
                                    dt_last = max(1, curr['frame'] - prev['frame']) / float(fps)
                                    maybe_update_auto_scale(dlast, dt_last)
                                    if meters_per_pixel is not None:
                                        xs = xs * float(meters_per_pixel)
                                        ys = ys * float(meters_per_pixel)
                                    else:
                                        xs = ys = None
                                if xs is not None and ys is not None:
                                    t0 = float(times.mean())
                                    t = times - t0
                                    if t.ptp() > 0:
                                        try:
                                            px = np.polyfit(t, xs, 1)
                                            py = np.polyfit(t, ys, 1)
                                            vx = float(px[0])
                                            vy = float(py[0])
                                            x_fit = px[0] * t + px[1]
                                            y_fit = py[0] * t + py[1]
                                            def _r2(y, yhat):
                                                ss_res = float(np.sum((y - yhat) ** 2))
                                                ss_tot = float(np.sum((y - y) .mean() ** 2))
                                                return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                                            # If fit is poor, fallback later
                                            speed_mps = float(np.hypot(vx, vy))
                                            speed_kph = speed_mps * 3.6 * speed_scale
                                        except Exception:
                                            pass

                            # Minimum threshold and EMA smoothing
                            if speed_kph < 8.0:
                                speed_kph = 0.0
                            prev = speed_ema.get(vehicle_id)
                            if prev is None:
                                smoothed = speed_kph
                            else:
                                smoothed = (1 - ema_alpha) * prev + ema_alpha * speed_kph
                            speed_ema[vehicle_id] = smoothed
                            if smoothed > max_speed_kph:
                                smoothed = max_speed_kph
                            speed = float(smoothed)
                            is_overspeed = speed > 60
                            
                            # Draw bounding box
                            color = (0, 0, 255) if is_overspeed else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw ID above, speed below to avoid overlap
                            cv2.putText(frame, f"ID:{vehicle_id}", (x1, max(10, y1-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            if speed > 0:
                                cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Store per-frame row only if we have smoothed speed
                            if speed > 0:
                                detection_data.append({
                                    'frame_number': frame_count,
                                    'vehicle_id': vehicle_id,
                                    'speed': round(speed, 2),
                                    'is_overspeed': is_overspeed,
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'confidence': round(confidence, 2)
                                })
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Save CSV
    if detection_data:
        df = pd.DataFrame(detection_data)
        df.to_csv(csv_path, index=False)
    
    print(f"Processing complete. Saved {len(detection_data)} detections")
    return len(detection_data)
