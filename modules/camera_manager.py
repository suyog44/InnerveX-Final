import cv2
from picamera2 import Picamera2

def list_cameras():
    info = Picamera2.global_camera_info()
    for cam in info:
        print("[CAM]", cam)
    return info

def pick_camera_num(info, model_name):
    model_name = model_name.lower()
    for cam in info:
        if model_name in str(cam).lower():
            return cam["Num"]
    return None

def build_preview_config(cam, size):
    cfg = cam.create_preview_configuration(
        main={"size": size, "format": "XBGR8888"},
        buffer_count=6
    )
    try:
        cfg.enable_raw(False)
    except Exception:
        pass
    return cfg

def open_dual_cameras(idx_master, idx_secondary, size):
    cam_master = Picamera2(camera_num=idx_master)
    cam_secondary = Picamera2(camera_num=idx_secondary)

    cam_master.configure(build_preview_config(cam_master, size))
    cam_secondary.configure(build_preview_config(cam_secondary, size))

    cam_master.start()
    cam_secondary.start()

    return cam_master, cam_secondary

def capture_bgr(cam):
    frame = cam.capture_array("main")
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
