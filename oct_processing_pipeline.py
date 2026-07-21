import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage import io


def load_oct_volume(tiff_path):
    """
    Loads a multi-page TIFF stack into a 3D Numpy array (Z, Y, X).
    """
    print(f"[1/5] Loading OCT stack from: {tiff_path}")
    volume = io.imread(tiff_path)

    # Ensure 3D array layout (Slices/Z, Height/Y, Width/X)
    if volume.ndim == 2:
        volume = np.expand_dims(volume, axis=0)

    # Normalize volume to 8-bit unsigned integer (0-255)
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8) * 255.0
    return volume.astype(np.uint8)


def filter_volume_3d(volume, kernel_size=(3, 5, 5)):
    """
    Applies a 3D Median Filter across the (Z, Y, X) dimensions to suppress
    OCT speckle noise while preserving inter-slice continuity along the Z-axis.
    """
    print(f"[2/5] Applying 3D Median Filter (Kernel: {kernel_size})...")
    filtered_vol = median_filter(volume, size=kernel_size)
    return filtered_vol


def detect_boundaries_2d(bscan_filtered):
    """
    Detects upper (air-tissue) and lower boundary transitions on a single 2D B-scan
    using vertical directional Sobel derivatives.
    """
    # 1. Compute Vertical Gradient (Sobel Y derivative)
    # Directional gradient helps distinguish air->tissue (positive) from tissue->air (negative)
    sobel_y = cv2.Sobel(bscan_filtered, cv2.CV_64F, dx=0, dy=1, ksize=5)

    # Gaussian blur the gradient to prevent noise spikes from breaking line continuity
    sobel_y_smooth = cv2.GaussianBlur(sobel_y, (5, 5), 0)

    height, width = bscan_filtered.shape
    top_boundary = np.zeros(width, dtype=int)
    bottom_boundary = np.zeros(width, dtype=int)

    # 2. Extract upper and lower surface profile across each A-scan column
    for x in range(width):
        column_grad = sobel_y_smooth[:, x]

        # Upper boundary: strongest positive transition (dark background to bright tissue)
        top_idx = np.argmax(column_grad)
        top_boundary[x] = top_idx if column_grad[top_idx] > 10 else 0

        # Lower boundary: strongest negative transition (bright tissue to dark deep region)
        # Only search below the upper boundary
        search_start = min(top_idx + 15, height - 1)
        if search_start < height - 1:
            bottom_idx = search_start + np.argmin(column_grad[search_start:])
            bottom_boundary[x] = bottom_idx if abs(column_grad[bottom_idx]) > 10 else height - 1
        else:
            bottom_boundary[x] = height - 1

    # Smooth boundary curves across adjacent columns (X-axis)
    top_boundary = cv2.GaussianBlur(top_boundary.astype(np.float32), (15, 1), 0).astype(int).ravel()
    bottom_boundary = cv2.GaussianBlur(bottom_boundary.astype(np.float32), (15, 1), 0).astype(int).ravel()

    return top_boundary, bottom_boundary


def create_binary_mask(shape, top_boundary, bottom_boundary):
    """
    Generates a solid 8-bit binary mask (White = Tissue, Black = Background)
    filling the region enclosed between the detected upper and lower boundaries.
    """
    height, width = shape
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        y_start = top_boundary[x]
        y_end = bottom_boundary[x]
        if y_end > y_start and y_start > 0:
            binary_mask[y_start:y_end, x] = 255

    # Apply 2D morphological Closing to fill small inner voids or gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return binary_mask


def create_overlay_image(bscan_raw, top_boundary, bottom_boundary):
    """
    Draws detected boundary lines over the RGB copy of the original B-scan.
    Green = Top Boundary (Surface), Red = Bottom Boundary
    """
    overlay = cv2.cvtColor(bscan_raw, cv2.COLOR_GRAY2BGR)
    width = bscan_raw.shape[1]

    for x in range(1, width):
        # Draw Top Boundary Line (Green)
        pt1_top = (x - 1, top_boundary[x - 1])
        pt2_top = (x, top_boundary[x])
        if pt1_top[1] > 0 and pt2_top[1] > 0:
            cv2.line(overlay, pt1_top, pt2_top, (0, 255, 0), 2)

        # Draw Bottom Boundary Line (Red)
        pt1_bot = (x - 1, bottom_boundary[x - 1])
        pt2_bot = (x, bottom_boundary[x])
        cv2.line(overlay, pt1_bot, pt2_bot, (0, 0, 255), 2)

    return overlay


def run_pipeline(tiff_input_path, output_dir="professor_review_output"):
    """
    Main execution pipeline. Processes the volume and exports four-panel comparison
    figures for professor evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    raw_volume = load_oct_volume(tiff_input_path)
    num_slices, height, width = raw_volume.shape

    # 2. Apply 3D Filtering across whole volume
    filtered_volume = filter_volume_3d(raw_volume, kernel_size=(3, 5, 5))

    print(f"[3/5] Extracting boundaries and binarizing {num_slices} B-scans...")

    # Process representative slices (e.g., 25%, 50%, and 75% through the stack)
    slices_to_export = [
        int(num_slices * 0.25),
        int(num_slices * 0.50),
        int(num_slices * 0.75)
    ]

    for slice_idx in range(num_slices):
        raw_bscan = raw_volume[slice_idx]
        filt_bscan = filtered_volume[slice_idx]

        # 3. Detect 2D Boundaries
        top_b, bot_b = detect_boundaries_2d(filt_bscan)

        # 4. Create Binary Mask & Overlay
        binary_mask = create_binary_mask((height, width), top_b, bot_b)
        overlay_img = create_overlay_image(raw_bscan, top_b, bot_b)

        # 5. Save visual report panels for selected key slices
        if slice_idx in slices_to_export:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(raw_bscan, cmap='gray')
            axes[0].set_title(f"1. Raw B-Scan (Slice {slice_idx})")
            axes[0].axis('off')

            axes[1].imshow(filt_bscan, cmap='gray')
            axes[1].set_title("2. 3D Filtered B-Scan")
            axes[1].axis('off')

            axes[2].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
            axes[2].set_title("3. Detected Boundaries\n(Green: Top | Red: Bottom)")
            axes[2].axis('off')

            axes[3].imshow(binary_mask, cmap='gray')
            axes[3].set_title("4. Final Binarized Mask")
            axes[3].axis('off')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"BScan_Review_Slice_{slice_idx:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f" -> Saved professor review figure: {save_path}")

    print(f"\n[5/5] Processing complete! Figures saved in folder: '{output_dir}/'")


if __name__ == "__main__":
    # Replace with the path to your multi-page .tiff stack file
    INPUT_TIFF_FILE = "Teeth11_2_RawBuffer7_Processed_Volume.tif"

    if os.path.exists(INPUT_TIFF_FILE):
        run_pipeline(INPUT_TIFF_FILE)
    else:
        print(
            f"Error: Could not find file '{INPUT_TIFF_FILE}'. Please update INPUT_TIFF_FILE variable with your stack path.")