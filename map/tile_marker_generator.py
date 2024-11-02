import os
import json
from math import pi, atan, sinh, degrees

# Directory where tiles are stored
tiles_directory = "map/server/tiles/"

# Convert tile coordinates to lat/lng based on zoom level
def tile_to_lat_lon(zoom, x, y):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = atan(sinh(pi * (1 - 2 * y / n)))
    lat_deg = degrees(lat_rad)
    return lat_deg, lon_deg

# Gather all tile coordinates and convert to lat/lon
tile_markers = []
for zoom_level in os.listdir(tiles_directory):
    zoom_dir = os.path.join(tiles_directory, zoom_level)
    if not os.path.isdir(zoom_dir):
        continue
    for x in os.listdir(zoom_dir):
        x_dir = os.path.join(zoom_dir, x)
        if not os.path.isdir(x_dir):
            continue
        for y_file in os.listdir(x_dir):
            if y_file.endswith(".png"):
                y = y_file.split(".")[0]
                lat, lon = tile_to_lat_lon(int(zoom_level), int(x), int(y))
                tile_markers.append({
                    "zoom": int(zoom_level),
                    "x": int(x),
                    "y": int(y),
                    "lat": lat,
                    "lon": lon
                })

# Save to a JSON file
with open("tile_markers.json", "w") as f:
    json.dump(tile_markers, f, indent=4)
