
import os
import threading
import queue
import time
import socket
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO

# ----------------------------------------------------------------------
# --------------------------- CONFIGURATION ---------------------------
# ----------------------------------------------------------------------
MODEL_PATH = "Gayan_set2.pt"          # <--- YOUR CUSTOM MODEL
CONF_THRESHOLD = 0.4                 # same as test.py
DETECTION_RESIZE = 640              # YOLO input size (square)
DETECT_EVERY_NFRAMES = 1
REMOVE_LOST = 3
MARKER_SIZE_MM = 100.0

DISPLAY_W = 480
DISPLAY_H = 360

EXTRA_ROTATION = {"triangle": 0, "circle": 0.0, "hexagon": 0}

# HSV colour ranges (green, red, black, white, blue)
COLOR_RANGES_HSV = {
    "green": ((35, 50, 50), (85, 255, 255)),
    "red1":  ((0, 120, 70), (10, 255, 255)),
    "red2":  ((170, 120, 70), (180, 255, 255)),
    "blue":  ((90, 50, 50), (130, 255, 255)),
    "white": ((0, 0, 200), (180, 40, 255)),
    "black": ((0, 0, 0), (180, 255, 45)),
}
COLOR_LABELS = ["green", "red", "black", "white", "blue"]
COLOR_OPTIONS = ["green", "red", "black", "white", "blue","unknown"]
SHAPE_MAP = {0: "triangle", 1: "circle", 2: "hexagon"}

# ----------------------------------------------------------------------
# --------------------------- GLOBALS ---------------------------------
# ----------------------------------------------------------------------
model = None
model_lock = threading.Lock()

frame_lock = threading.Lock()
latest_frame = None
frame_counter = 0

detected_objects_lock = threading.Lock()
detected_objects: Dict[int, "DetectedObject"] = {}
next_object_id = 1

comm_out_queue = queue.Queue()
comm_in_queue = queue.Queue()

camera_thread = None
camera_stop_event = threading.Event()
use_file_source = False
video_file_path = None
full_auto_mode = False

# ----------------------------------------------------------------------
# --------------------------- DATA CLASSES ----------------------------
# ----------------------------------------------------------------------
@dataclass
class RotBox:
    cls: int
    score: float
    x1: float; y1: float
    x2: float; y2: float
    x3: float; y3: float
    x4: float; y4: float

@dataclass
class DetectedObject:
    id: int
    shape: int
    shape_name: str
    color: str
    seen_count: int = 0
    lost_count: int = 0
    last_seen_frame: int = 0
    pixel_coords: Tuple[float, float] = (0.0, 0.0)
    mm_coords: Tuple[float, float] = (0.0, 0.0)
    min_rect: any = None
    min_angle: float = 0.0
    seen_flag: bool = True
    shape_angle: float = 0.0

# ----------------------------------------------------------------------
# --------------------------- MODEL -----------------------------------
# ----------------------------------------------------------------------
def load_yolo_model() -> YOLO:
    global model
    if model is not None:
        return model
    print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return None
    try:
        model = YOLO(MODEL_PATH)
        print("[INFO] Model loaded")
    except Exception as e:
        print(f"[ERROR] YOLO init failed: {e}")
        model = None
    return model

# ----------------------------------------------------------------------
# --------------------------- LETTERBOX -------------------------------
# ----------------------------------------------------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad to new_shape while keeping aspect ratio."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    if dw or dh:
        resized = cv2.copyMakeBorder(resized, dh, dh, dw, dw,
                                     cv2.BORDER_CONSTANT, value=color)
    return resized, r, (dw, dh)

# ----------------------------------------------------------------------
# --------------------------- ARUCO CALIBRATION -----------------------
# ----------------------------------------------------------------------
class Calibration:
    def __init__(self):
        self.valid = False
        self.px_per_mm = 0.0
        self.origin_px = (0.0, 0.0)
        self.unit_vec_x = np.array([1.0, 0.0])
        self.unit_vec_y = np.array([0.0, 1.0])

calib = Calibration()
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
detector = cv2.aruco.ArucoDetector(ARUCO_DICT)

def detect_aruco_and_calibrate(img):
    global calib
    if calib.valid:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return
    c = corners[0].reshape((4, 2)).astype(np.float32)
    origin = c[0]
    vec_x = c[3] - c[0]
    vec_y = c[1] - c[0]
    len_x = np.linalg.norm(vec_x)
    len_y = np.linalg.norm(vec_y)
    if len_x == 0 or len_y == 0:
        return
    unit_x = vec_x / len_x
    unit_y = vec_y - np.dot(vec_y, unit_x) * unit_x
    unit_y /= np.linalg.norm(unit_y)
    px_per_mm = (len_x + len_y) / (2 * MARKER_SIZE_MM)

    calib.valid = True
    calib.px_per_mm = px_per_mm
    calib.origin_px = origin
    calib.unit_vec_x = unit_x
    calib.unit_vec_y = unit_y
    print(f"[CALIB] px_per_mm = {px_per_mm:.2f}")

def px_to_mm(px_x, px_y):
    if not calib.valid:
        return 0.0, 0.0
    pt = np.array([px_x, px_y])
    rel = pt - calib.origin_px
    mm_x = np.dot(rel, calib.unit_vec_x) / calib.px_per_mm
    mm_y = np.dot(rel, calib.unit_vec_y) / calib.px_per_mm
    return float(mm_x), float(mm_y)

# ----------------------------------------------------------------------
# --------------------------- COLOR -----------------------------------
# ----------------------------------------------------------------------
def classify_color_at_point(img, x, y, radius=6):
    h, w = img.shape[:2]
    x = int(round(x))
    y = int(round(y))
    x0 = max(0, x - radius)
    x1 = min(w - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(h - 1, y + radius)
    
    patch = img[y0:y1+1, x0:x1+1]
    if patch.size == 0:
        return "unknown"

    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        if not np.isfinite(mean_hsv).all():
            return "unknown"
        H, S, V = [int(x) for x in mean_hsv.flatten()]  # <--- FIX HERE
    except Exception as e:
        print(f"[ERROR] classify_color_at_point: {e}")
        return "unknown"

    for label in COLOR_LABELS:
        if label == "red":
            r1, r2 = COLOR_RANGES_HSV["red1"], COLOR_RANGES_HSV["red2"]
            in_r1 = (r1[0][0] <= H <= r1[1][0] and
                     r1[0][1] <= S <= r1[1][1] and
                     r1[0][2] <= V <= r1[1][2])
            in_r2 = (r2[0][0] <= H <= r2[1][0] and
                     r2[0][1] <= S <= r2[1][1] and
                     r2[0][2] <= V <= r2[1][2])
            if in_r1 or in_r2:
                return "red"
        else:
            mn, mx = COLOR_RANGES_HSV[label]
            if (mn[0] <= H <= mx[0] and
                mn[1] <= S <= mx[1] and
                mn[2] <= V <= mx[2]):
                return label
    return "unknown"

# ----------------------------------------------------------------------
# --------------------------- YOLO DETECTION -------------------------
# ----------------------------------------------------------------------
last_inference_time = 0.0

def run_detector(resized_img) -> List[RotBox]:
    global model, last_inference_time
    
    with model_lock:
        if model is None:
            load_yolo_model()

        
        results = model.predict(resized_img,
                                conf=CONF_THRESHOLD,
                                iou=0.45,
                                device='cpu',
                                verbose=False)
        resultt = results[0]
        
        last_inference_time = resultt.speed['inference']

    boxes = []
    for r in results:
        # === Try OBB first ===
        if hasattr(r, "obb") and r.obb is not None and len(r.obb.cls) > 0:
            obb = r.obb
            for i in range(len(obb.cls)):
                if obb.conf[i] < CONF_THRESHOLD:
                    continue
                c = obb.xyxyxyxy[i].cpu().numpy().flatten()
                boxes.append(RotBox(
                    cls=int(obb.cls[i]),
                    score=float(obb.conf[i]),
                    x1=c[0], y1=c[1],
                    x2=c[2], y2=c[3],
                    x3=c[4], y3=c[5],
                    x4=c[6], y4=c[7]
                ))
            print(f"[YOLO] OBB: {len(boxes)} detections")
            continue  # prefer OBB if available

        # === Fallback: Standard boxes (convert to 4-corner) ===
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes.cls) > 0:
            std = r.boxes
            for i in range(len(std.cls)):
                if std.conf[i] < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = std.xyxy[i].cpu().numpy()
                # Convert axis-aligned box to 4 corners
                boxes.append(RotBox(
                    cls=int(std.cls[i]),
                    score=float(std.conf[i]),
                    x1=x1, y1=y1,
                    x2=x2, y2=y1,
                    x3=x2, y3=y2,
                    x4=x1, y4=y2
                ))
            #print(f"[YOLO] STD: {len(boxes)} detections")

    #print(f"[YOLO] TOTAL: {len(boxes)} det | {last_inference_time:.3f}s")
    return boxes


def polygon_from_contour(contour, eps_factor=0.02):
    """Return approximated polygon points (Nx2 int) from contour."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
    pts = approx.reshape((-1, 2)).astype(int)
    return pts

def edges_from_polygon(pts):
    """Return list of edges: [(x1,y1,x2,y2,length,angle_deg), ...]"""
    edges = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))  # -180..180
        # normalize angle to 0..180 for undirected edge reference
        edges.append((x1, y1, x2, y2, length, angle))
    return edges



def choose_reference_edge(edges):
    """Return the edge whose direction is closest to horizontal."""
    if not edges:
        return None
    # e[5] = normalized angle 0-180
    return max(edges, key=lambda e: min(e[5], 180 - e[5]))
# ----------------------------------------------------------------------
# --------------------------- PROCESS DETECTIONS ----------------------
# ----------------------------------------------------------------------
def process_detections(dets_resized, scale_info, orig_img, frame_idx,app):
    global next_object_id
    h0, w0 = orig_img.shape[:2]
    r, (dw, dh) = scale_info

    buffer = 0
    with detected_objects_lock:
        for obj in detected_objects.values():
            obj.seen_flag = False

        for det in dets_resized:
            # ---- back-scale to original resolution ----
            pts = np.array([
                [(det.x1 - dw) / r, (det.y1 - dh) / r],
                [(det.x2 - dw) / r, (det.y2 - dh) / r],
                [(det.x3 - dw) / r, (det.y3 - dh) / r],
                [(det.x4 - dw) / r, (det.y4 - dh) / r]
            ], dtype=np.float32)

            x_min = max(int(np.min(pts[:, 0]) - buffer), 0)
            x_max = min(int(np.max(pts[:, 0]) + buffer), w0 - 1)
            y_min = max(int(np.min(pts[:, 1]) - buffer), 0)
            y_max = min(int(np.max(pts[:, 1]) + buffer), h0 - 1)

            roi = orig_img[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #blur = cv2.bilateralFilter(gray, 9, 75, 75)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                continue
            cx_rel = int(M["m10"] / M["m00"])
            cy_rel = int(M["m01"] / M["m00"])

            cx = cx_rel + x_min
            cy = cy_rel + y_min

            min_rect = cv2.minAreaRect(largest)

            (min_rect_cx_rel, min_rect_cy_rel), (w, h), angle = min_rect
            
            min_rect_cx_rel = min_rect_cx_rel + x_min
            min_rect_cy_rel = min_rect_cy_rel + y_min

            

            color = classify_color_at_point(orig_img, cx, cy)
            allowed_colors = [c for c, v in app.color_vars.items() if v.get()]
            if color not in allowed_colors:
                continue  # skip this detection

            mmx, mmy = px_to_mm(cx, cy)
            shape_name = SHAPE_MAP.get(det.cls, "unknown")

            pts = polygon_from_contour(largest)         
            pts_img = pts + np.array([x_min, y_min])

            edges = edges_from_polygon(pts_img)
            ref = choose_reference_edge(edges)

            if ref is not None:
                x1, y1, x2, y2,lengh,raw_angle= ref
                
                # raw_angle is the *real* direction of the base (0-180)
                dx = x2 - x1
                dy = y2 - y1

                # Normal vector (perpendicular to the line)
                if dx == 0 and dy == 0:
                        px, py = x1, y1
                        nx_v = px - cx
                        ny_v = py - cy
                                   


                t = ((cx - x1) * dx + (cy - y1) * dy) / (dx*dx + dy*dy)
                px = x1 + t * dx
                py = y1 + t * dy

                    # Vector from (cx,cy) to projection (this is the perpendicular we want)
                nx_v = px - cx
                ny_v = py - cy

                    # Angle of this normal vector
                # --- Convert to ArUco-based angle ---
                if calib.valid:
                    v_img = np.array([nx_v, ny_v], dtype=np.float32)
                    v_img /= np.linalg.norm(v_img) + 1e-9  # normalize safely

                    vx_local = np.dot(v_img, calib.unit_vec_x)
                    vy_local = np.dot(v_img, calib.unit_vec_y)

                    shape_angle = math.degrees(math.atan2(vy_local, vx_local))
                    if shape_angle < 0:
                        shape_angle += 360.0
                else:
                    # fallback if ArUco not calibrated yet
                    shape_angle = math.degrees(math.atan2(ny_v, nx_v))
                    if shape_angle < 0:
                        shape_angle += 360.0

            # ---- tracking (simple nearest-neighbour) ----
            matched_id = None
            for oid, obj in detected_objects.items():
                dx = obj.pixel_coords[0] - cx
                dy = obj.pixel_coords[1] - cy
                if math.hypot(dx, dy) < max(w, h) * 0.6 and obj.color == color:
                    matched_id = oid
                    break

            if matched_id is None:
                oid = next_object_id
                next_object_id += 1
                detected_objects[oid] = DetectedObject(
                    id=oid,
                    shape=det.cls,
                    shape_name=shape_name,
                    color=color,
                    seen_count=1,
                    last_seen_frame=frame_idx,
                    pixel_coords=(min_rect_cx_rel, min_rect_cy_rel),
                    mm_coords=(mmx, mmy),
                    min_rect=((min_rect_cx_rel, min_rect_cy_rel), (w, h), angle),
                    min_angle=angle,
                    shape_angle=shape_angle
                )
            else:
                obj = detected_objects[matched_id]
                obj.seen_count += 1
                obj.lost_count = 0
                obj.last_seen_frame = frame_idx
                obj.pixel_coords = (cx, cy)
                obj.mm_coords = (mmx, mmy)
                obj.min_rect = ((min_rect_cx_rel, min_rect_cy_rel), (w, h), angle)
                obj.min_angle = angle
                obj.color = color
                obj.shape = det.cls
                obj.shape_name = shape_name
                obj.shape_angle=shape_angle

        # ---- remove lost objects ----
        to_del = [oid for oid, obj in detected_objects.items()
                  if frame_idx - obj.last_seen_frame > REMOVE_LOST]
        for oid in to_del:
            del detected_objects[oid]

# ----------------------------------------------------------------------
# --------------------------- CAMERA THREAD ---------------------------
# ----------------------------------------------------------------------
def camera_thread_fn(ready_event):
    global latest_frame, frame_counter, use_file_source, video_file_path
    from pypylon import pylon

    if use_file_source:
        while (not video_file_path or not os.path.isfile(video_file_path)) and not camera_stop_event.is_set():
            time.sleep(0.5)
        if camera_stop_event.is_set():
            return
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video {video_file_path}")
            ready_event.set()
            return
        print(f"[INFO] Playing video: {video_file_path}")
        ready_event.set()
        while not camera_stop_event.is_set():
            ret, f = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with frame_lock:
                latest_frame = f.copy()
                frame_counter += 1
            time.sleep(0.01)
        cap.release()
        return

    # ----- Pylon camera -----
    try:
        tl = pylon.TlFactory.GetInstance()
        devs = tl.EnumerateDevices()
        if not devs:
            print("[ERROR] No Pylon camera")
            ready_event.set()
            return
        cam = pylon.InstantCamera(tl.CreateFirstDevice())
        cam.Open()
        cam.ExposureAuto.SetValue('Off')      # disable auto exposure
        cam.ExposureTime.SetValue(10000.0)  
        conv = pylon.ImageFormatConverter()
        conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print("[INFO] Pylon camera started")
        ready_event.set()
        while not camera_stop_event.is_set() and cam.IsGrabbing():
            res = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if res.GrabSucceeded():
                img = conv.Convert(res).GetArray()
                with frame_lock:
                    latest_frame = img.copy()
                    frame_counter += 1
            res.Release()
            time.sleep(0.01)
        cam.StopGrabbing()
        cam.Close()
    except Exception as e:
        print(f"[ERROR] Camera thread: {e}")

# ----------------------------------------------------------------------
# --------------------------- COMM THREAD -----------------------------
# ----------------------------------------------------------------------
def comm_thread_fn(host='192.168.125.201', port=5000):
    """Handle communication with ABB robot."""
    global full_auto_mode
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((host, port))
        server.listen(1)
        print(f"Comm server listening on {host}:{port}")
    except OSError as e:
        print(f"[ERROR] Cannot bind to {host}:{port}: {e}")
        return

    try:
        conn, addr = server.accept()
        print("Robot connected from", addr)
        conn.settimeout(1.0)

        while True:
            try:
                data = conn.recv(4094)
            except socket.timeout:
                data = b''

            if data:
                s = data.decode("latin-1").strip()
                print("Robot ->", s)

                if s.lower().startswith("ready"):
                    # Try to get next queued message
                    try:
                        msg = comm_out_queue.get_nowait()
                    except queue.Empty:
                        msg = None

                    # If no message queued but full-auto is active, send the first object
                    if msg is None and full_auto_mode:
                        with detected_objects_lock:
                            objs = list(detected_objects.values())
                        if objs:
                            obj = objs[0]
                            mx, my = obj.mm_coords
                            rot = obj.shape_angle + EXTRA_ROTATION.get(obj.shape_name, 0.0)
                            msg = f"{obj.shape}/{mx:.1f},{my:.1f}#{rot:.1f}"
                            print(f"[AUTO READY] -> {msg}")

                    # Now actually send the message if we have one
                    if msg:
                        conn.sendall((msg).encode("UTF-8"))
                        print("Sent ->", msg)


                elif s.lower().startswith("done") and full_auto_mode:
                    with detected_objects_lock:
                        objs = list(detected_objects.values())
                    if objs:
                      obj_id = list(detected_objects.keys())[0]
                      obj = detected_objects[obj_id]
                      mx, my = obj.mm_coords
                      rot = obj.shape_angle + EXTRA_ROTATION.get(obj.shape, 0.0)
                      msg = f"{obj.shape}/{mx:.1f},{my:.1f}#{rot:.1f}"
                      conn.sendall((msg).encode("UTF-8"))
                      print(f"[FULL AUTO] Sent -> {msg}")

            # Remove it from memory so it won't be reused
                      del detected_objects[obj_id]
                      print(f"[INFO] Removed object ID {obj_id} after pickup")

                    else:
                      print("[FULL AUTO] No objects detected.")
                      full_auto_mode = False

                            
                else:
                    comm_in_queue.put(s)

            try:
                out = comm_out_queue.get_nowait()
                conn.sendall((out).encode("UTF-8"))
            except queue.Empty:
                pass

            time.sleep(0.05)

    except Exception as e:
        print("Comm thread error:", e)
    finally:
        conn.close()
        server.close()

# ----------------------------------------------------------------------
# --------------------------- GUI ------------------------------------
# ----------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Pick & Place")

        # ---- left canvas (video) ----
        self.left = tk.Frame(root, width=DISPLAY_W, height=DISPLAY_H)
        self.left.grid(row=0, column=0, sticky='nswe')
        self.canvas = tk.Canvas(self.left, bg="black",
                                width=DISPLAY_W, height=DISPLAY_H)
        self.canvas.pack(fill="both", expand=True)

        # ---- right panel ----
        right = tk.Frame(root)
        right.grid(row=0, column=1, sticky='nswe')

        # >>> Make both panels resize properly <<<
        root.grid_columnconfigure(0, weight=3)  # video side
        root.grid_columnconfigure(1, weight=2)  # right side
        root.grid_rowconfigure(0, weight=1)

        # ---- source selection ----
        src = ttk.LabelFrame(right, text="Source")
        src.pack(fill="x", padx=5, pady=5)
        self.src_var = tk.StringVar(value="camera")
        tk.Radiobutton(src, text="Camera", variable=self.src_var,
                       value="camera", command=self._switch_source).pack(anchor="w")
        tk.Radiobutton(src, text="Video file", variable=self.src_var,
                       value="file", command=self._switch_source).pack(anchor="w")
        tk.Button(src, text="Choose file", command=self._choose_file).pack(fill="x")

        # ---- color filter ----
        color_frame = ttk.LabelFrame(right, text="Color Filter")
        color_frame.pack(fill="x", padx=5, pady=5)

        self.color_vars = {}
        for color in COLOR_OPTIONS:
            var = tk.BooleanVar(value=True)  # default = show all
            self.color_vars[color] = var
            cb = tk.Checkbutton(color_frame, text=color.capitalize(), variable=var, command=self._refresh_color_selection)
            cb.pack(anchor="w")
        
        # Add a manual update button
        tk.Button(color_frame, text="Update Colors", command=self._update_colors).pack(fill="x", pady=5)


        # ---- controls ----
        ctrl = ttk.LabelFrame(right, text="Controls")
        ctrl.pack(fill="x", padx=5, pady=5)
        self.mode_var = tk.StringVar(value="full")
        tk.Radiobutton(ctrl, text="Full auto", variable=self.mode_var,
                       value="full").pack(anchor="w")
        tk.Radiobutton(ctrl, text="Single object", variable=self.mode_var,
                       value="single").pack(anchor="w")
        self.start_btn = tk.Button(ctrl, text="Start", command=self._start)
        self.stop_btn = tk.Button(ctrl, text="Stop", command=self._stop)
        self.start_btn.pack(side="left", padx=5, pady=5)
        self.stop_btn.pack(side="left", padx=5, pady=5)


        # ---- detected objects list ----
        mid = ttk.LabelFrame(right, text="Detected Objects")
        mid.pack(fill="both", expand=True, padx=5, pady=5)
        self.listbox = tk.Listbox(mid, height=12)
        self.listbox.pack(fill="both", expand=True)

        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        

        # ---- status ----
        self.status = ttk.Label(right, text="YOLO: 0.000 s")
        self.status.pack(fill="x", padx=5, pady=2)

        # ---- schedule updates ----
        self._schedule_update()
        self._refresh_color_selection()

    def _refresh_color_selection(self):
        """Force initial sync of color selections."""
        global app_instance
        allowed_colors = [c for c, v in self.color_vars.items() if v.get()]

    def _update_colors(self):
        """Manually update allowed colors based on checkboxes."""
        allowed_colors = [c for c, v in self.color_vars.items() if v.get()]   

    def _switch_source(self):
        global use_file_source
        use_file_source = self.src_var.get() == "file"
        self._restart_camera()

    def _choose_file(self):
        global video_file_path, use_file_source
        p = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All", "*.*")])
        if p:
            video_file_path = p
            use_file_source = True
            self.src_var.set("file")
            self._restart_camera()

    def _restart_camera(self):
        global camera_thread, camera_stop_event
        camera_stop_event.set()
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=1.0)
        camera_stop_event.clear()
        ready = threading.Event()
        camera_thread = threading.Thread(target=camera_thread_fn,
                                         args=(ready,), daemon=True)
        camera_thread.start()
        ready.wait(timeout=5.0)

    def _start(self):
        if self.mode_var.get() == "full":
            global full_auto_mode
            full_auto_mode = True
            with detected_objects_lock:
                if detected_objects:
                    obj = next(iter(detected_objects.values()))
                    mx, my = obj.mm_coords
                    rot = obj.shape_angle + EXTRA_ROTATION.get(obj.shape_name, 0.0)
                    msg = f"{obj.shape}/{mx:.1f},{my:.1f}#{rot:.1f}"
                    comm_out_queue.put(msg)
                    print(f"[INIT] → {msg}")

    def _on_select(self, event):
        if self.mode_var.get() != "single":
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        oid = list(detected_objects.keys())[sel[0]]
        obj = detected_objects[oid]
        mx, my = obj.mm_coords
        rot = obj.shape_angle + EXTRA_ROTATION.get(obj.shape_name, 0.0)
        msg = f"{obj.shape}/{mx:.1f},{my:.1f}#{rot:.1f}"
        comm_out_queue.put(msg)
        print(f"[SINGLE SELECT] → {msg}")

    def _stop(self):
        comm_out_queue.put("STOP")

    # ------------------------------------------------------------------
    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        with detected_objects_lock:
            for oid, obj in detected_objects.items():
                txt = (f"ID:{oid}  {obj.shape_name}  {obj.color}  "
                       f"mm=({obj.mm_coords[0]:.1f},{obj.mm_coords[1]:.1f})  "
                       f"angle={obj.shape_angle:.1f}°")
                self.listbox.insert(tk.END, txt)

    # ------------------------------------------------------------------
    def _draw_frame(self, frame):
        disp = frame.copy()

        # ---- calibration axes (if any) ----
        if calib.valid:
            ox, oy = map(int, calib.origin_px)
            length = int(calib.px_per_mm * MARKER_SIZE_MM)
            ex = ox + int(length * calib.unit_vec_x[0])
            ey = oy + int(length * calib.unit_vec_x[1])
            cv2.arrowedLine(disp, (ox, oy), (ex, ey), (0, 0, 255), 2)   # X red
            ex = ox + int(length * calib.unit_vec_y[0])
            ey = oy + int(length * calib.unit_vec_y[1])
            cv2.arrowedLine(disp, (ox, oy), (ex, ey), (0, 255, 0), 2)   # Y green

        # ---- draw every tracked object ----
        with detected_objects_lock:
            for obj in detected_objects.values():
                cx, cy = map(int, obj.pixel_coords)
                # center point
                cv2.circle(disp, (cx, cy), 10, (0, 0, 255), -1)
                # orientation arrow
                angle_radians = np.deg2rad(obj.shape_angle)
                x2 = int(cx + 100 * np.cos(angle_radians))
                y2 = int(cy + 100 * np.sin(angle_radians))
                cv2.arrowedLine(disp,(cx, cy),(x2,y2),(0, 255, 0),10,line_type=cv2.LINE_AA,tipLength=0.1)               
                # yellow rect
                box = cv2.boxPoints(obj.min_rect).astype(int)
                cv2.polylines(disp, [box], True, (0, 255, 255), 2)          
                txt = f"{obj.shape_name}/{obj.color}"
                cv2.putText(disp, txt, (cx + 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ---- fit to canvas ----
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = DISPLAY_W, DISPLAY_H
        h, w = disp.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        disp_res = cv2.resize(disp, (new_w, new_h))

        # centre on canvas
        x_off = (canvas_w - new_w) // 2
        y_off = (canvas_h - new_h) // 2

        img_rgb = cv2.cvtColor(disp_res, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil)

        # keep reference
        self.canvas.tk_img = tk_img
        if hasattr(self, "img_id"):
            self.canvas.itemconfig(self.img_id, image=tk_img)
            self.canvas.coords(self.img_id, x_off, y_off)
        else:
            self.img_id = self.canvas.create_image(x_off, y_off,
                                                   anchor="nw", image=tk_img)

    # ------------------------------------------------------------------
    def _schedule_update(self):

        self._update()
        self.root.after(50, self._schedule_update)

    def _update(self):
        
        global latest_frame, frame_counter
        with frame_lock:
            f = latest_frame.copy() if latest_frame is not None else None
            fc = frame_counter

        if f is None:
            return

        # ---- calibration (once) ----
        detect_aruco_and_calibrate(f)

        # ---- inference (every N frames) ----
        if fc % DETECT_EVERY_NFRAMES == 0:
            resized, r, pad = letterbox(f, (DETECTION_RESIZE, DETECTION_RESIZE))
            dets = run_detector(resized)
            process_detections(dets, (r, pad), f, fc,self)
            self._refresh_list()

        self.status.config(text=f"YOLO: {last_inference_time:.3f}s")
        self._draw_frame(f)
        self._refresh_color_selection()

# ----------------------------------------------------------------------
# --------------------------- MAIN ------------------------------------
# ----------------------------------------------------------------------
def main():
    # Load model **once**
    load_yolo_model()

    # Camera thread
    ready = threading.Event()
    global camera_thread
    camera_thread = threading.Thread(target=camera_thread_fn,
                                     args=(ready,), daemon=True)
    camera_thread.start()
    ready.wait(timeout=5.0)

    # Comm thread
    threading.Thread(target=comm_thread_fn, daemon=True).start()
    

    root = tk.Tk()

    global app_instance
    app_instance = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
