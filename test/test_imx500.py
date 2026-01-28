from picamera2 import Picamera2
import time

def open_imx500_preview(cam_index, preferred_sizes=((640,640),(640,480),(1280,720))):
    cam = Picamera2(camera_num=cam_index)

    # Print modes (optional but useful)
    print(f"\n[IMX500] Camera {cam_index} sensor modes:")
    for i, m in enumerate(cam.sensor_modes):
        print(f"  [{i}] {m}")

    # Try preferred preview sizes in order
    for sz in preferred_sizes:
        try:
            cfg = cam.create_preview_configuration(
                main={"size": sz, "format": "XBGR8888"},
                buffer_count=8
            )
            # Disable raw stream if API exists (recommended) [1](https://www.waveshare.com/wiki/MLX90640-D110_Thermal_Camera)
            try:
                cfg.enable_raw(False)
            except Exception:
                try:
                    cfg["raw"] = None
                except Exception:
                    pass

            cam.configure(cfg)
            cam.start()
            time.sleep(0.3)

            # Confirm final config
            final = cam.camera_configuration()
            print("[IMX500] Started with main config:", final["main"])
            return cam, sz

        except Exception as e:
            print(f"[IMX500] Failed preview size {sz}: {e}")
            try:
                cam.stop()
            except Exception:
                pass

    cam.close()
    raise RuntimeError("IMX500 could not start with preferred preview sizes.")
