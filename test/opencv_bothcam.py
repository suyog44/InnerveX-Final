
#!/usr/bin/env python3
import time
from datetime import datetime
import cv2
from picamera2 import Picamera2

def start_camera(camera_index, size=(1280, 720), buffer_count=4):
    """
    Create and start a Picamera2 instance for the given camera index.
    Returns (picam2, name) where name is a readable label.
    """
    p = Picamera2(camera_num=camera_index)
    # Choose a sensible preview size per sensor; adjust as needed
    config = p.create_preview_configuration(main={"size": size}, buffer_count=buffer_count)
    p.configure(config)
    p.start()
    # Make a readable name string
    name = f"CAM{camera_index}"
    return p, name

def put_fps(frame, label, fps):
    cv2.putText(frame, f"{label}  {fps:.1f} FPS",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)

def main():
    # NOTE: From your list, 0: imx500, 1: imx519
    cam0, name0 = start_camera(0, (1280, 720))   # IMX500
    cam1, name1 = start_camera(1, (1280, 720))   # IMX519

    # FPS counters
    t0 = t1 = time.time()
    n0 = n1 = 0
    fps0 = fps1 = 0.0

    print("[q] quit  [1] save IMX500 frame  [2] save IMX519 frame")

    try:
        while True:
            # Grab frames
            f0 = cam0.capture_array()   # BGR
            f1 = cam1.capture_array()

            # Update FPS every ~1s
            n0 += 1; now = time.time()
            if now - t0 >= 1.0:
                fps0 = n0 / (now - t0); n0 = 0; t0 = now
            n1 += 1
            if now - t1 >= 1.0:
                fps1 = n1 / (now - t1); n1 = 0; t1 = now

            put_fps(f0, f"{name0} (IMX500)", fps0)
            put_fps(f1, f"{name1} (IMX519)", fps1)

            cv2.imshow("IMX500 (CAM0)", f0)
            cv2.imshow("IMX519 (CAM1)", f1)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('1'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"imx500_{ts}.png"
                cv2.imwrite(fn, f0)
                print("Saved", fn)
            elif k == ord('2'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"imx519_{ts}.png"
                cv2.imwrite(fn, f1)
                print("Saved", fn)

    finally:
        cam0.stop(); cam1.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
