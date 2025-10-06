"""
Filename: real_time_super_resolution.py
Description: Real-time super-resolution for thermal imaging using MLX90640 and Pi Camera 2 motion tracking.
Team Members:
- Yiwei Wang (yiweiwan@umich.edu)
- Yufei Xi (yufeixi@umich.edu)
- Carissa Wu (carissawu@umich.edu)
License: For educational use, unrestricted for teaching.
"""

import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import cv2
from picamera2 import Picamera2
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# === Initialize SR Viewer ===
# === Load and preprocess data ===

# === Utility functions ===
def clip_to_grid(dx_thermal, dy_thermal, resolution):
    step = 1.0 / resolution
    return step * round(dx_thermal / step), step * round(dy_thermal / step)

def transform_translation(dx, dy,
                          src_res=(640, 480), src_fov=(62.2, 48.8),
                          dst_res=(32, 24), dst_fov=(110, 75)):
    scale_x = (dst_res[0] / src_res[0]) * (dst_fov[0] / src_fov[0])
    scale_y = (dst_res[1] / src_res[1]) * (dst_fov[1] / src_fov[1])
    return dx * scale_x, dy * scale_y

# === Viewer class ===
class RealTimeThermalViewer:
    def __init__(self, shape_needed=(48, 64), buffer_size=4, subpixel_size=0.5, interp_method="bilinear"):
        self.buffer_size = buffer_size
        self.shape_needed = shape_needed
        self.subpixel_size = subpixel_size
        self.interp_method = interp_method
        self.frames = []
        self.translations = []

        self.fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        self.ax_orig, self.ax_up, self.ax_sr = axs
        self.fig.subplots_adjust(top=0.85)
        self.text = self.fig.text(0.5, 0.92, "", ha='center', va='top', fontsize=12)

        empty = np.zeros((24, 32), dtype=np.uint8)
        up = self.upscale(empty)
        sr = np.zeros(self.shape_needed, dtype=np.uint8)

        self.img_orig = self.ax_orig.imshow(empty, cmap='inferno', vmin=0, vmax=255)
        self.img_up = self.ax_up.imshow(up, cmap='inferno', vmin=0, vmax=255)
        self.img_sr = self.ax_sr.imshow(sr, cmap='inferno', vmin=0, vmax=255)

        self.ax_orig.set_title("Original (24×32)")
        self.ax_up.set_title(f"Upscaled ({self.interp_method})")
        self.ax_sr.set_title(f"Super Resolution ({self.interp_method})")

        plt.ion()
        plt.show()

    def get_interp(self):
        return {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "lanczos": cv2.INTER_LANCZOS4
        }.get(self.interp_method, cv2.INTER_NEAREST)

    def upscale(self, image):
        return cv2.resize(np.uint8(image), (self.shape_needed[1], self.shape_needed[0]), interpolation=self.get_interp())

    def super_resolve(self):
        canvas = np.zeros(self.shape_needed, dtype=np.float32)
        count = np.zeros(self.shape_needed, dtype=np.float32)
        upscale_factor = (self.shape_needed[0] // 24, self.shape_needed[1] // 32)
        interp = self.get_interp()

        for frame, (dx, dy) in zip(self.frames, self.translations):
            M = np.float32([[1, 0, dx * upscale_factor[1]], [0, 1, dy * upscale_factor[0]]])
            upscaled = cv2.resize(frame, (self.shape_needed[1], self.shape_needed[0]), interpolation=interp)
            shifted = cv2.warpAffine(upscaled, M, (self.shape_needed[1], self.shape_needed[0]), flags=interp,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            canvas += shifted
            count += (shifted > 0).astype(np.float32)

        count[count == 0] = 1
        return np.clip(np.round(canvas / count), 0, 255).astype(np.uint8)

    def step(self, frame, translation):
        self.frames.append(frame)
        self.translations.append(transform_translation(*translation))
        if len(self.frames) > self.buffer_size:
            self.frames.pop(0)
            self.translations.pop(0)

        self.img_orig.set_data(frame)
        self.img_up.set_data(self.upscale(frame))

        if len(self.frames) == self.buffer_size:
            sr = self.super_resolve()
            self.img_sr.set_data(sr)

            dx, dy = translation
            dx_th, dy_th = transform_translation(dx, dy)
            dx_clip, dy_clip = clip_to_grid(dx_th, dy_th, resolution=1 / self.subpixel_size)
            self.text.set_text(
                f"Frame {len(self.frames)} | PiCam dx: {dx:.2f}, dy: {dy:.2f} | "
                f"Thermal dx: {dx_th:.2f}, dy: {dy_th:.2f} | "
                f"Rounded: dx = {dx_clip:.2f}, dy = {dy_clip:.2f}"
            )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

# === Run viewer with delay T ===
viewer = RealTimeThermalViewer(shape_needed=(48, 64), buffer_size=4, interp_method="bilinear")




# === Initialize Thermal Camera (MLX90640) ===
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
mlx_shape = (24, 32)
thermal_frame = np.zeros((24 * 32,))
frame_size = (320, 240)

# === Initialize Raspberry Pi Camera 2 ===
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

# === Optical Flow Parameters ===
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_frame = picam2.capture_array()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

def show_frame_with_translation(thermal_frame, translation):
    global viewer
    viewer.step(thermal_frame, translation)

print("Started. Press 'q' to quit.")

try:
    while True:
        # === Capture Thermal Frame ===
        try:
            mlx.getFrame(thermal_frame)
            data_array = np.reshape(thermal_frame, mlx_shape)
            norm_data = cv2.normalize(data_array, None, 0, 255, cv2.NORM_MINMAX)
            thermal_img = np.round(norm_data).astype(np.uint8)
            #norm_data = np.uint8(norm_data)
            #resized_img = cv2.resize(norm_data, frame_size, interpolation=cv2.INTER_CUBIC)
            #thermal_img = cv2.applyColorMap(resized_img, cv2.COLORMAP_INFERNO)
        except ValueError:
            print("Thermal sensor not ready, sleeping...")
            time.sleep(0.1)
            continue

        # === Capture RGB Frame & Calculate Optical Flow ===
        curr_frame = picam2.capture_array()
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)

        if prev_pts is None or len(prev_pts) < 5:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
            dx, dy = 0.0, 0.0
        else:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
            if next_pts is not None:
                valid_prev_pts = prev_pts[status == 1]
                valid_next_pts = next_pts[status == 1]
                motion = valid_next_pts - valid_prev_pts
                dx, dy = np.mean(motion, axis=0) if len(motion) > 0 else (0.0, 0.0)
                prev_pts = valid_next_pts.reshape(-1, 1, 2)
            else:
                dx, dy = 0.0, 0.0
                prev_pts = None

        prev_gray = curr_gray.copy()

        # === Show and Print ===
        show_frame_with_translation(thermal_img, [dx, dy])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

cv2.destroyAllWindows()
