import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import xarray as xr
import sys
#infile = sys.argv[1]
# data = pd.read_csv("zhenghetrack.csv")
# fig_buoys = px.scatter_map(data, lat="Latitude", lon="Longitude")
# fig = go.Figure(data = fig_buoys)
# fig.update_layout(mapbox_style="open-street-map")
# fig.show()


# alternative
aruco_file = "test2.nc"
ds_aruco = xr.open_dataset(aruco_file)

# Last 48 hours relative to the final timestamp in the file
end_time = ds_aruco.position_time.values[-1]
start_time = end_time - np.timedelta64(48, "h")

pos = ds_aruco[["lon", "lat"]].sel(position_time=slice(start_time, end_time))
# t_hours = (imu.time.values - imu.time.values[0]).astype("float64") / 1e9 / 3600