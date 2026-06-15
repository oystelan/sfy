import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ARUCO TEST: plot IMU data from the last 48 hours
aruco_file = "test.nc"

ds_aruco = xr.open_dataset(aruco_file)

# Last 48 hours relative to the final timestamp in the file
end_time = ds_aruco.time.values[-1]
start_time = end_time - np.timedelta64(48, "h")

imu = ds_aruco[["w_x", "w_y", "w_z"]].sel(time=slice(start_time, end_time))
# t_hours = (imu.time.values - imu.time.values[0]).astype("float64") / 1e9 / 3600

print(f"File: {aruco_file}")
print(f"Full file time range: {ds_aruco.time.values[0]} to {ds_aruco.time.values[-1]}")
print(f"Selected last-48h window: {start_time} to {end_time}")
print(f"Samples plotted: {imu.sizes['time']}")
print(f"Frequency: {ds_aruco.attrs.get('frequency')} Hz")

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axes[0].plot(imu.time.values, imu.w_x, lw=0.8, color="black")
axes[0].set_ylabel("w_x")
axes[0].grid(True, alpha=0.3)

axes[1].plot(imu.time.values, imu.w_y, lw=0.8, color="tab:green")
axes[1].set_ylabel("w_y")
axes[1].grid(True, alpha=0.3)

axes[2].plot(imu.time.values, imu.w_z, lw=0.8, color="tab:red")
axes[2].set_ylabel("w_z")
axes[2].set_xlabel("Time since selected window start (hours)")
axes[2].grid(True, alpha=0.3)

fig.suptitle("arucoTest.nc IMU data, last 48 hours in file", fontsize=12)
plt.tight_layout()
plt.show()
