"""
Filename: offline_super_resolution.py
Description: Offline super-resolution processing of thermal frames and translation data for image enhancement.
Team Members:
- Yiwei Wang (yiweiwan@umich.edu)
- Yufei Xi (yufeixi@umich.edu)
- Carissa Wu (carissawu@umich.edu)
License: For educational use, unrestricted for teaching.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# === Load and preprocess data ===
thermal_df = pd.read_csv("./thermal_raw_data.csv")
translation_df = pd.read_csv("./translation_data.csv")

thermal_frames = thermal_df.iloc[:, 1:].to_numpy().reshape(-1, 24, 32)
thermal_frames = cv2.normalize(thermal_frames, None, 0, 255, cv2.NORM_MINMAX)
thermal_frames = np.round(thermal_frames).astype(np.uint8)
translations = translation_df[["dx", "dy"]].to_numpy()

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
T = 0.1  # seconds between updates
viewer = RealTimeThermalViewer(shape_needed=(48, 64), buffer_size=4, interp_method="bilinear")

for i in range(len(thermal_frames)):
    viewer.step(thermal_frames[i], translations[i])
    time.sleep(T)
