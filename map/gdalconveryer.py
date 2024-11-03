from osgeo import gdal, osr
from PIL import Image
import os
import subprocess

# Enable GDAL exceptions for better error handling
gdal.UseExceptions()

# Replace with the path to your image
image_path = "result.png"
image = Image.open(image_path)
width, height = image.size

print(f"Image width: {width}, height: {height}")

# Define the output paths
georeferenced_image_path = "./map/server/photos/georeferenced_image.tif"
warped_image_path = "./map/server/photos/warped_image.tif"
tiles_output_path = "./map/server/photos/tiles/"
tiles_output_path_dir = "./map/server/photos/tiles/"

# Define the geographic coordinates for each corner of the image
top_left = (39.8303, 32.7125)     # Sol üst köşe
top_right = (39.8300, 32.7136)    # Sağ üst köşe
bottom_left = (39.8311, 32.7464)  # Sol alt köşe
bottom_right = (39.8397, 32.7464) # Sağ alt köşe

# Step 1: Georeference the image using GDAL Python bindings
# Step 1: Georeference the image using GDAL Python bindings
def georeference_image(input_image_path, output_image_path, width, height):
    print("Georeferencing the image...")

    # Open the image as a GDAL dataset
    dataset = gdal.Open(input_image_path)
    driver = gdal.GetDriverByName('GTiff')
    georeferenced_dataset = driver.CreateCopy(output_image_path, dataset, 0)

    # Define GCPs (ground control points) with positive latitude values
    gcp_list = [
        gdal.GCP(top_left[1], abs(top_left[0]), 0, 0, 0),
        gdal.GCP(top_right[1], abs(top_right[0]), 0, width, 0),
        gdal.GCP(bottom_left[1], abs(bottom_left[0]), 0, 0, height),
        gdal.GCP(bottom_right[1], abs(bottom_right[0]), 0, width, height)
    ]

    # Set spatial reference to WGS84 (EPSG:4326)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)

    # Apply GCPs with spatial reference
    georeferenced_dataset.SetGCPs(gcp_list, spatial_ref.ExportToWkt())

    # Close the dataset
    georeferenced_dataset = None
    print("Georeferencing completed.")


# Step 2: Warp the image to Web Mercator projection (EPSG:3857)
def warp_image(input_image_path, output_image_path):
    print("Warping the image to Web Mercator projection (EPSG:3857)...")
    gdal.Warp(output_image_path, input_image_path, dstSRS="EPSG:3857")
    print("Warping completed.")

# Step 3: Generate map tiles using gdal2tiles.py
def generate_tiles(input_image_path, output_dir, min_zoom=15, max_zoom=19):
    print("Generating map tiles...")

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Call gdal2tiles.py using subprocess with desired tile options
    command = [
        ".\.venv\Scripts\gdal2tiles.exe",            # Assumes gdal2tiles.py is in PATH
        "-z", f"{min_zoom}-{max_zoom}",  # Zoom levels
        input_image_path,
        output_dir
    ]
    
    subprocess.run(command, check=True)
    print("Tile generation completed.")

# Run all steps
if __name__ == "__main__":
    georeference_image(image_path, georeferenced_image_path, width, height)
    warp_image(georeferenced_image_path, warped_image_path)
    generate_tiles(warped_image_path, tiles_output_path)
