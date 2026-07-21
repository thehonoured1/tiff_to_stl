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

    # Ensure 3D array layout if 2D (Slices/Z, Height/Y, Width/X)
    if volume.ndim == 2:
        volume = np.expand_dims(volume, axis=0)

    # Normalize volume to 8-bit unsigned integer (0-255)
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8) * 255.0
    return volume.astype(np.uint8)


def filter_volume_3d(volume, kernel_size=(3, 3, 3)):
    """
    Applies a 3D Median Filter across the (Z, Y, X) dimensions to suppress
    OCT speckle noise while preserving inter-slice continuity along the Z-axis.
        PARAMETER TUNING:
        Higher Z (e.g., 5, 5, 5): Increase if you have many slices close together and need stronger smoothing across adjacent B-scans.
        Lower Z (e.g., 1, 5, 5): Use if you have very few B-scans (large slice spacing along Z) to avoid blurring different cross-sections into each other.
        Higher Y/X (e.g., 3, 7, 7): Increase if your raw images are extremely grainy or have heavy speckle noise.
    """
    print(f"[2/5] Applying 3D Median Filter (Kernel: {kernel_size})...")
    filtered_vol = median_filter(volume, size=kernel_size)  #kernel size == window size
    return filtered_vol


def graph_cut_solver(cost_matrix, min_threshold, max_step=20):
    """
    Solves for the globally optimal continuous surface path using Dynamic Programming (Graph-Cut).
    Prevents vertical zigzag jumps while respecting minimum gradient intensity thresholds.
    """
    height, width = cost_matrix.shape
    dp = np.full((height, width), np.inf, dtype=np.float32) #store the cumulative cost of the cheapest path to reach any given pixel
    backtrack = np.zeros((height, width), dtype=int)        #prev Y coordinates.

    # Initialize first column
    dp[:, 0] = cost_matrix[:, 0]

    # Forward Accumulation Pass
    for x in range(1, width):
        for y in range(height):
            y_min = max(0, y - max_step)            #python's min/max returns the min/max value among its params.
            y_max = min(height, y + max_step + 1)   #y_min/max works to restrict range of checking. y as the current pixel, you can only fluctuate +- max_step

            prev_costs = dp[y_min:y_max, x - 1]     #    x-1 refers to the prev row.
            best_prev_offset = np.argmin(prev_costs)    # returns the index (position) of the minimum value along a specified axis
            best_prev_y = y_min + best_prev_offset

            dp[y, x] = cost_matrix[y, x] + dp[best_prev_y, x - 1]
            backtrack[y, x] = best_prev_y

    # Backtracking Pass
    optimal_path = np.zeros(width, dtype=int) # numpy array declaration.
    optimal_path[-1] = np.argmin(dp[:, -1]) #in python, : means entire, -1 refers to the last value.

    for x in range(width - 1, 0, -1):
        optimal_path[x - 1] = backtrack[optimal_path[x], x]

    # Validate path against user's minimum intensity threshold
    for x in range(width):
        best_y = optimal_path[x]
        if (np.max(cost_matrix[:, x]) - cost_matrix[best_y, x]) < min_threshold:
            optimal_path[x] = 0 if min_threshold == 2 else height - 1

    return optimal_path


def detect_boundaries_2d(bscan_filtered):
    """
    Detects upper (air-tissue) and lower boundary transitions on a single 2D B-scan
    using Method 3 (Graph-Cut / Dynamic Programming) on vertical directional Sobel derivatives.
        PARAMETER TUNING:
        ksize=3: Picks up fine, sharp detail. Ideal for high-resolution scans where the enamel surface transition occurs over just a few pixels.
        ksize=7: Averages over a wider vertical band. Ideal for blurry, low-contrast tissue transitions (like soft gum tissue) or lower-resolution scans.
    """
    # 1. Compute Vertical Gradient (Sobel Y derivative)
    sobel_y = cv2.Sobel(bscan_filtered, cv2.CV_64F, dx=0, dy=1, ksize=7)
                                                                #   ▲ must be odd
    # Gaussian blur the gradient to prevent noise spikes from breaking line continuity
    sobel_y_smooth = cv2.GaussianBlur(sobel_y, (3, 3), 0)

    height, width = bscan_filtered.shape

    # PARAMETER TUNING:
    # Upper boundary: strongest positive transition (dark background to bright tissue)
    # --- METHOD 3: UPPER BOUNDARY GRAPH-CUT ---
    max_pos_grad = np.max(sobel_y_smooth)
    top_cost = max_pos_grad - sobel_y_smooth # invert the high gradient so it appears 'cheap' to graph_cut_solver.
    top_cost = np.clip(top_cost, 0, None)

    # Reduced threshold to 2 so faint enamel reflections are recognized
    top_boundary = graph_cut_solver(top_cost, min_threshold=2, max_step=50)
    #                                                       ▲
    #                                                       └── Min gradient intensity

    # PARAMETER TUNING: Smooth boundary curves across adjacent columns (X-axis)
    # Reduced kernel to (5, 1) so steep peaks are not flattened out
    top_boundary = cv2.GaussianBlur(top_boundary.astype(np.float32), (5, 1), 0).astype(int).ravel()
                #                                                           ▲
                #                                                           └── Horizontal smoothing window (must be odd)
    return top_boundary


def create_binary_mask(shape, top_boundary, offset=10):
    height, width = shape
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        y_start = top_boundary[x]
        y_end = y_start + offset  # Fixed 5px thickness 📐
        if y_start > 0:
            binary_mask[y_start:y_end, x] = 255 # create the 5 pixels of white.
    # gap cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                #                           ^dilation followed by an erosion. fills in tiny single-pixel gaps or holes inside the white band.

def create_overlay_image(bscan_raw, top_boundary, offset=5):
    overlay = cv2.cvtColor(bscan_raw, cv2.COLOR_GRAY2BGR)
    width = bscan_raw.shape[1]

    for x in range(1, width):
        # Top Boundary Line (Green) 🟢
        pt1_top = (x - 1, top_boundary[x - 1])
        pt2_top = (x, top_boundary[x])
        if pt1_top[1] > 0 and pt2_top[1] > 0:
            cv2.line(overlay, pt1_top, pt2_top, (0, 255, 0), 2) # drawing

        # 5px Offset Line (Red) 🔴
        pt1_bot = (x - 1, top_boundary[x - 1] + offset)
        pt2_bot = (x, top_boundary[x] + offset)
        cv2.line(overlay, pt1_bot, pt2_bot, (0, 0, 255), 2)

    return overlay


def run_pipeline(tiff_input_path, output_dir="professor_review_output", demo_mode=True):
    """
    Main execution pipeline. Processes the volume and exports four-panel comparison
    figures for professor evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    raw_volume = load_oct_volume(tiff_input_path)
    num_slices, height, width = raw_volume.shape

    # 2. Apply 3D Filtering across whole volume
    filtered_volume = filter_volume_3d(raw_volume, kernel_size=(3, 3, 3))

    # Representative slices to inspect
    slices_to_export = [
        int(num_slices * 0.25),
        int(num_slices * 0.50),
        int(num_slices * 0.75)
    ]
    if 250 < num_slices and 250 not in slices_to_export:
        slices_to_export.append(250)

    slices_to_process = slices_to_export if demo_mode else range(num_slices)

    print(f"[3/5] DEMO MODE: Extracting boundaries for {len(slices_to_process)} key B-scans...")

    for slice_idx in slices_to_process:
        raw_bscan = raw_volume[slice_idx]
        filt_bscan = filtered_volume[slice_idx]

        # 3. Detect 2D Upper Boundary Only
        top_b = detect_boundaries_2d(filt_bscan)

        # 4. Create Binary Mask & Overlay (5px offset)
        binary_mask = create_binary_mask((height, width), top_b, offset=10)
        overlay_img = create_overlay_image(raw_bscan, top_b, offset=10)

        # 5. Save visual report panels for selected key slices
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(raw_bscan, cmap='gray')
        axes[0].set_title(f"1. Raw B-Scan (Slice {slice_idx})")
        axes[0].axis('off')

        axes[1].imshow(filt_bscan, cmap='gray')
        axes[1].set_title("2. 3D Filtered B-Scan")
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title("3. Graph-Cut Boundaries\n(Green: Top | Red: Bottom)")
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
    INPUT_TIFF_FILE = "Teeth11_2_RawBuffer7_Processed_Volume.tif"

    if os.path.exists(INPUT_TIFF_FILE):
        run_pipeline(INPUT_TIFF_FILE, demo_mode=True)
    else:
        print(
            f"Error: Could not find file '{INPUT_TIFF_FILE}'. Please update INPUT_TIFF_FILE variable with your stack path.")