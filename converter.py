# Teeth34_2_RawBuffer5_Processed_Volume.tif
# Teeth31_2_RawBuffer6_Processed_Volume.tif
# Teeth11_2_RawBuffer7_Processed_Volume.tif
import numpy as np
from skimage import io, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_opening, disk  # Added for 2D cleaning
from scipy.ndimage import label  # Added for 3D island removal
from stl import mesh

print("Loading volume...")
volume = io.imread('Teeth11_2_RawBuffer7_Processed_Volume.tif')

# 1. Re-optimized Downsampling (Lightweight Z, Detailed X and Y)
volume = volume[::2, ::2, ::2]

print("Applying fast edge-preserving smoothing...")
smoothed_volume = gaussian(volume, sigma=1.0, preserve_range=True)

print("Calculating threshold...")
optimal_threshold = threshold_otsu(smoothed_volume)

# 2. Thresholding
binary_volume = smoothed_volume > (optimal_threshold * 1.15)

# --- NEW STEP: 2D Morphological Cleaning ---
print("Cleaning background noise slice-by-slice...")
# A small radius disk (1 or 2) will erase the isolated spray pixels
# without altering the overall geometry of the tooth.
footprint = disk(1)
for z in range(binary_volume.shape[0]):
    binary_volume[z] = binary_opening(binary_volume[z], footprint)

# --- NEW STEP: 3D Connected Components (Keep Largest Component) ---
print("Isolating main 3D tooth structure (removing floating shards)...")
# Label distinct 3D objects. Background is 0.
labeled_volume, num_features = label(binary_volume)
if num_features > 1:
    # Count the size of each labeled feature (ignoring background index 0)
    component_sizes = np.bincount(labeled_volume.ravel())
    component_sizes[0] = 0  # Do not choose the background as the largest component

    # Keep only the largest contiguous 3D object (the tooth structure)
    largest_component_label = np.argmax(component_sizes)
    binary_volume = (labeled_volume == largest_component_label)
    print(f"Removed {num_features - 1} floating background artifacts!")

# Pad boundaries to close the mesh windows cleanly
binary_volume = np.pad(binary_volume, pad_width=2, mode='constant', constant_values=False)

print("Extracting 3D surface...")
# 3. Run marching cubes
verts, faces, normals, values = measure.marching_cubes(binary_volume.astype(float), level=0.5)
print(f"Generated a detailed mesh with {len(faces)} triangles.")

# 4. Fast Vectorized Mapping
print("Packing into STL...")
tooth_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
tooth_mesh.vectors = verts[faces]

# 5. Save
tooth_mesh.save('output_tooth.stl')
print("Successfully generated detailed output_tooth.stl!")