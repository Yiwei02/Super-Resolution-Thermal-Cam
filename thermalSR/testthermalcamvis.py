"""
Filename: thermal_camera_visualizer.py
Description: Simple real-time thermal frame visualization using MLX90640 on Raspberry Pi.
Team Members:
- Yiwei Wang (yiweiwan@umich.edu)
- Yufei Xi (yufeixi@umich.edu)
- Carissa Wu (carissawu@umich.edu)
License: For educational use, unrestricted for teaching.
"""
##########################################
# MLX90640 Thermal Camera w Raspberry Pi
# -- 2Hz Sampling with Simple Routine
##########################################
#
import time,board,busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
# import cv2

i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ # set refresh rate
mlx_shape = (24,32)

# setup the figure for plotting
plt.ion() # enables interactive plotting
fig,ax = plt.subplots(figsize=(12,7))
therm1 = ax.imshow(np.zeros(mlx_shape),vmin=0,vmax=32,cmap="jet") #start plot with zeros
#cbar = fig.colorbar(therm1) # setup colorbar for temps
#cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14) # colorbar label

frame = np.zeros((24*32,)) # setup array for storing all 768 temperatures
t_array = []
while True:
    t1 = time.monotonic()
    try:
        mlx.getFrame(frame) # read MLX temperatures into frame var
        #frame = cv2.flip(frame,1)
        data_array = (np.reshape(frame,mlx_shape)) # reshape to 24x32
        therm1.set_data((data_array)) # flip left to right
        therm1.set_clim(vmin=0,vmax=50) # set bounds
       # cbar.on_mappable_changed(therm1) # update colorbar range
        plt.pause(0.001) # required
#        fig.savefig('mlx90640_test_fliplr.png',dpi=300,facecolor='#FCFCFC',
#                    bbox_inches='tight') # comment out to speed up
        t_array.append(time.monotonic()-t1)
        print('Sample Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))
    except ValueError:
        continue # if error, just read again
