import cv2
import numpy as np
import os
import sys


clicked_points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(param, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(param, f"{len(clicked_points)}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Click 4+ ground points (press W when done)', param)


def load_image_from_video(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to open video: ' + video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0:
        frame_index = 0
    if frame_index >= total:
        frame_index = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Failed to read frame from video')
    return frame


def main():
    print('Calibration helper - creates homography.npy in project root')
    print('Usage:')
    print('  python calibrate_homography.py <image_path>')
    print('  or')
    print('  python calibrate_homography.py --video <video_path> [--frame 100]')

    img = None
    args = sys.argv[1:]
    if not args:
        print('No arguments provided. Attempting to use the first frame of a default video in media/videos if present...')
        default_dir = os.path.join(os.path.dirname(__file__), 'media', 'videos')
        if os.path.isdir(default_dir):
            candidates = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if candidates:
                img = load_image_from_video(candidates[0], 0)
        if img is None:
            print('Please provide an image or video path.')
            return
    else:
        if args[0] == '--video':
            if len(args) < 2:
                print('Please provide a video path after --video')
                return
            video_path = args[1]
            frame_index = 0
            if '--frame' in args:
                try:
                    idx = args.index('--frame')
                    frame_index = int(args[idx + 1])
                except Exception:
                    pass
            img = load_image_from_video(video_path, frame_index)
        else:
            image_path = args[0]
            img = cv2.imread(image_path)
            if img is None:
                print('Failed to read image:', image_path)
                return

    draw = img.copy()
    cv2.namedWindow('Click 4+ ground points (press W when done)', cv2.WINDOW_NORMAL)
    cv2.imshow('Click 4+ ground points (press W when done)', draw)
    cv2.setMouseCallback('Click 4+ ground points (press W when done)', on_mouse, draw)

    print('Instructions:')
    print('- Click at least 4 points that lie on the road plane (ground).')
    print('- Use stable features: lane line intersections, crosswalk corners, manhole covers, etc.')
    print('- When finished selecting points, press W to continue.')

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('w'), ord('W')):
            break
        if key in (27,):  # ESC to abort
            cv2.destroyAllWindows()
            print('Aborted')
            return

    cv2.destroyAllWindows()

    if len(clicked_points) < 4:
        print('Need at least 4 points. You clicked', len(clicked_points))
        return

    print('\nYou clicked these pixel points (x, y):')
    for i, p in enumerate(clicked_points, 1):
        print(f'{i}: {p[0]} {p[1]}')

    print('\nFor each pixel point, enter real-world coordinates in meters (X Y).')
    print('Tip: choose a simple local coordinate system. Example: set the first point to 0 0,')
    print('then use known road distances along roughly orthogonal directions for others.')

    world_pts = []
    for i in range(len(clicked_points)):
        while True:
            try:
                raw = input(f'World coords for point {i+1} (format: X Y): ').strip()
                X, Y = map(float, raw.split())
                world_pts.append((X, Y))
                break
            except Exception:
                print('Invalid input. Please enter two numbers like: 12.3 4.56')

    pixel_pts = np.array(clicked_points, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)

    H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC)
    if H is None:
        print('Failed to compute homography. Try different points.')
        return

    project_root = os.path.dirname(__file__)
    out_path = os.path.join(project_root, 'homography.npy')
    np.save(out_path, H.astype(np.float32))
    print('Saved homography to:', out_path)

    # Quick sanity check: map first pixel point
    pt = np.array([[clicked_points[0]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)[0][0]
    print('Sanity check: first pixel point maps to world ~', mapped)


if __name__ == '__main__':
    main()


