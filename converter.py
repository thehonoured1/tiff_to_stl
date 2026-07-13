# Teeth34_2_RawBuffer5_Processed_Volume.tif
# Teeth31_2_RawBuffer6_Processed_Volume.tif
# Teeth11_2_RawBuffer7_Processed_Volume.tif
import numpy as np
from skimage import io, measure
from skimage.filters import gaussian, threshold_local
from skimage.morphology import binary_opening, binary_closing, disk, ball, remove_small_objects
from scipy.ndimage import label, binary_fill_holes
from stl import mesh

print("Loading volume...")
volume = io.imread('Teeth31_2_RawBuffer6_Processed_Volume.tif')

# 1. Re-optimized Downsampling
volume = volume[::2, ::2, ::2]

# DIRECTIONAL CROP: Chop off the problematic far-left side completely
volume = volume[:, 0:, :]

print("Applying fast edge-preserving smoothing...")
smoothed_volume = gaussian(volume, sigma=1, preserve_range=True)

# --- REPLACED OTSU WITH LOCAL ADAPTIVE CONTRAST THRESHOLDING ---
print("Calculating local contrast thresholds slice-by-slice...")
binary_volume = np.zeros_like(smoothed_volume, dtype=bool)

# We loop through slices and calculate thresholds based on local pixel neighborhoods
for z in range(smoothed_volume.shape[0]):
    # block_size must be an odd number. It defines the search window.
    # offset subtracts a constant to kill off uniform background fog.
    local_thresh = threshold_local(smoothed_volume[z], block_size=101, offset=3)
    binary_volume[z] = smoothed_volume[z] > local_thresh


# Voxel Border Clearing ---
print("Clearing scan frame border artifacts...")
margin = 2
binary_volume[:margin, :, :] = False
binary_volume[-margin:, :, :] = False
binary_volume[:, :margin, :] = False
binary_volume[:, -margin:, :] = False
binary_volume[:, :, :margin] = False
binary_volume[:, :, -margin:] = False

# 3. 2D Morphological Cleaning
print("Cleaning background noise slice-by-slice...")
footprint = disk(2)
for z in range(binary_volume.shape[0]):
    binary_volume[z] = binary_opening(binary_volume[z], footprint)

# 4. 3D Cleaning Sequence
print("Cutting thin contiguous pixel bridges...")
binary_volume = binary_opening(binary_volume, ball(2))

print("Vaporizing remaining isolated noise masses...")
binary_volume = remove_small_objects(binary_volume, min_size=5) #OBSERVATION: doesn't affect final stl.

print("Isolating main 3D tooth structure...")
labeled_volume, num_features = label(binary_volume)
if num_features > 1:
    component_sizes = np.bincount(labeled_volume.ravel())
    component_sizes[0] = 0
    largest_component_label = np.argmax(component_sizes)
    binary_volume = (labeled_volume == largest_component_label)
    print(f"Successfully isolated tooth!")

print("Shrink-wrapping the hollow shell to bridge surface gaps...")
binary_volume = binary_closing(binary_volume, ball(3))

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