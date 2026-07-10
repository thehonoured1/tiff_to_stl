# Teeth34_2_RawBuffer5_Processed_Volume.tif
# Teeth31_2_RawBuffer6_Processed_Volume.tif
# Teeth11_2_RawBuffer7_Processed_Volume.tif
import numpy as np
from skimage import io, measure
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_opening, disk
from scipy.ndimage import label, binary_fill_holes  # Added binary_fill_holes
from stl import mesh

print("Loading volume...")
volume = io.imread('Teeth11_2_RawBuffer7_Processed_Volume.tif')

# 1. Re-optimized Downsampling
volume = volume[::2, ::2, ::2]

# DIRECTIONAL CROP: Chop off the problematic far-left side completely
# Adjust the '100' depending on how many slices deep that block goes
volume = volume[:, 0:, :]
#                  ^ slices the top of the tooth.

# Gaussian blur produces a top edge. Leave sigma=0.
print("Applying fast edge-preserving smoothing...")
smoothed_volume = gaussian(volume, sigma=0, preserve_range=True)

print("Calculating threshold...")
optimal_threshold = threshold_otsu(smoothed_volume)

# 2. Thresholding - Dialed down slightly from 1.15 to prevent punching holes
binary_volume = smoothed_volume > (optimal_threshold * 1.13)

# --- NEW STEP: Voxel Border Clearing ---
print("Clearing scan frame border artifacts...")
# Set a 5-voxel safety margin on the outer edges of the 3D volume
# This forcefully disconnects boundary artifacts/walls from the tooth structure.
margin = 5
binary_volume[:margin, :, :] = False
binary_volume[-margin:, :, :] = False
binary_volume[:, :margin, :] = False
binary_volume[:, -margin:, :] = False
binary_volume[:, :, :margin] = False
binary_volume[:, :, -margin:] = False

# 3. 2D Morphological Cleaning
print("Cleaning background noise slice-by-slice...")
footprint = disk(1)
for z in range(binary_volume.shape[0]):
    binary_volume[z] = binary_opening(binary_volume[z], footprint)

# --- NEW STEP: 3D Hole Filling ---
print("Filling internal gaping holes...")
# This fixes the hollow spaces inside the tooth mesh
binary_volume = binary_fill_holes(binary_volume)

# 4. 3D Connected Components (Keep Largest Component)
print("Isolating main 3D tooth structure...")
labeled_volume, num_features = label(binary_volume)
if num_features > 1:
    component_sizes = np.bincount(labeled_volume.ravel())
    component_sizes[0] = 0
    largest_component_label = np.argmax(component_sizes)
    binary_volume = (labeled_volume == largest_component_label)
    print(f"Successfully isolated tooth and dropped boundary artifacts!")

# Pad boundaries to close the mesh windows cleanly
binary_volume = np.pad(binary_volume, pad_width=2, mode='constant', constant_values=False)

print("Extracting 3D surface...")
verts, faces, normals, values = measure.marching_cubes(binary_volume.astype(float), level=0.5)
print(f"Generated a detailed mesh with {len(faces)} triangles.")

print("Packing into STL...")
tooth_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
tooth_mesh.vectors = verts[faces]

# 5. Save
tooth_mesh.save('output_tooth_clean.stl')
print("Successfully generated final clean output_tooth_clean.stl!")