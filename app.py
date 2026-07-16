import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from ultralytics import YOLO
import torch
from torchvision.ops import nms as torch_nms
import os
from PIL import Image

# Set up page config
st.set_page_config(
    page_title="Lunar Crater Detection & Safe Landing Site Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.title("Lunar Crater Detection & Safe Landing Site Analyzer 🚀")
st.markdown("""
This application detects lunar craters using a custom-trained **YOLOv8** model, analyzes crater density 
and hazards, and recommends the safest landing sites for lunar landers or rovers.
""")

# -------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------------------------------
WEIGHTS_PATH = "best-3.pt"  # Place your weights file in the same directory as this script

# Define 5 sample images (Ensure these paths exist in your repo/local directory)
SAMPLE_IMAGES = {
    "Select a Sample Image": None,
    "Sample 1: Lunar Mare Region": "samples/sample_1.png",
    "Sample 2: Crater Highlands": "samples/sample_2.png",
    "Sample 3: Impact Basin Edge": "samples/sample_3.png",
    "Sample 4: High-Resolution Survey A": "samples/sample_4.png",
    "Sample 5: High-Resolution Survey B": "samples/sample_5.png"
}

# -------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------
st.sidebar.header("🔧 Configuration Parameters")

# Sidebar parameter adjustments
PIXELS_PER_KM = st.sidebar.number_input("Pixels per Kilometer (Scale)", min_value=1.0, value=10.0, step=0.5)
WINDOW_SIZE = st.sidebar.slider("Sliding Window Size (pixels)", min_value=320, max_value=1280, value=640, step=320)
OVERLAP = st.sidebar.slider("Window Overlap Fraction", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
CONF_THRESHOLD = st.sidebar.slider("Detection Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
GRID_N = st.sidebar.slider("Grid Divisions (N x N)", min_value=4, max_value=16, value=8, step=1)
SAFETY_THRESHOLD = st.sidebar.slider("Hazard/Safety Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

# -------------------------------------------------------
# HELPER CORE LOGIC FUNCTIONS (Unmodified Processing Steps)
# -------------------------------------------------------

@st.cache_resource
def load_model(weights_path):
    """Loads the YOLO model and caches it to optimize app performance."""
    if not os.path.exists(weights_path):
        st.error(f"❌ Model weights not found at `{weights_path}`. Please verify your repository setup.")
        return None
    return YOLO(weights_path)

def get_window_positions(image_h, image_w, window_size, overlap):
    """Computes all sliding window positions."""
    step = int(window_size * (1 - overlap))
    
    y_starts = list(range(0, image_h - window_size + 1, step))
    if not y_starts or y_starts[-1] + window_size < image_h:
        y_starts.append(max(0, image_h - window_size))

    x_starts = list(range(0, image_w - window_size + 1, step))
    if not x_starts or x_starts[-1] + window_size < image_w:
        x_starts.append(max(0, image_w - window_size))

    return [(x, y) for y in y_starts for x in x_starts]

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Performs duplicate box removal using PyTorch's Non-Maximum Suppression."""
    if len(boxes) == 0:
        return [], []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = torch_nms(boxes_t, scores_t, iou_threshold)
    kept_boxes = boxes_t[keep].numpy().tolist()
    kept_scores = scores_t[keep].numpy().tolist()
    return kept_boxes, kept_scores

# Load Model
model = load_model(WEIGHTS_PATH)

# -------------------------------------------------------
# IMAGE INPUT SELECTION
# -------------------------------------------------------
st.subheader("📸 Choose Lunar Image Source")
upload_tab, sample_tab = st.tabs(["📤 Upload Your Own Image", "🖼️ Try Sample Images"])

uploaded_file = None
sample_selection = None
selected_img_path = None

with upload_tab:
    uploaded_file = st.file_uploader("Upload a Lunar Surface Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

with sample_tab:
    sample_selection = st.selectbox("Select one of our preset sample images:", list(SAMPLE_IMAGES.keys()))
    if SAMPLE_IMAGES[sample_selection] is not None:
        selected_img_path = SAMPLE_IMAGES[sample_selection]

# Process Image Retrieval
img_bgr = None
if uploaded_file is not None:
    # Convert Streamlit upload to openCV BGR image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif selected_img_path is not None:
    if os.path.exists(selected_img_path):
        img_bgr = cv2.imread(selected_img_path)
    else:
        st.error(f"Sample image `{selected_img_path}` not found in repository directories.")

# -------------------------------------------------------
# PIPELINE EXECUTION
# -------------------------------------------------------
if img_bgr is not None and model is not None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_bgr.shape[:2]
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Input Lunar Image", use_container_width=True)
    with col2:
        img_w_km = img_w / PIXELS_PER_KM
        img_h_km = img_h / PIXELS_PER_KM
        st.markdown(f"""
        ### 📊 Image Metadata
        - **Dimensions:** {img_w} x {img_h} pixels
        - **Scale Mapping:** {img_w_km:.2f} km x {img_h_km:.2f} km
        - **Scanning Grid:** {GRID_N} x {GRID_N} cells
        """)

    # 1. Slide and Predict
    st.write("---")
    st.info("🔍 Running sliding window object detection...")
    positions = get_window_positions(img_h, img_w, WINDOW_SIZE, OVERLAP)
    
    all_boxes = []
    all_scores = []
    
    progress_bar = st.progress(0.0)
    for i, (x, y) in enumerate(positions):
        patch = img_bgr[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
        results = model(patch, conf=CONF_THRESHOLD, verbose=False)
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                
                # Offset local coordinates to global coordinate system
                boxes[:, 0] += x
                boxes[:, 2] += x
                boxes[:, 1] += y
                boxes[:, 3] += y
                
                all_boxes.extend(boxes.tolist())
                all_scores.extend(scores.tolist())
                
        progress_bar.progress((i + 1) / len(positions))
    
    st.success(f"Total raw detections identified: **{len(all_boxes)}**")

    # 2. Non-Maximum Suppression (NMS)
    st.info("⚡ Applying Non-Maximum Suppression (NMS)...")
    kept_boxes, kept_scores = apply_nms(all_boxes, all_scores, iou_threshold=0.3)
    st.success(f"Detections retained after NMS filtering: **{len(kept_boxes)}**")

    if len(kept_boxes) > 0:
        # Convert to DataFrame
        df = pd.DataFrame(kept_boxes, columns=['x1', 'y1', 'x2', 'y2'])
        df['confidence'] = kept_scores
        df['center_x_px'] = (df['x1'] + df['x2']) / 2
        df['center_y_px'] = (df['y1'] + df['y2']) / 2
        df['width_px'] = df['x2'] - df['x1']
        df['height_px'] = df['y2'] - df['y1']

        # Convert to km
        df['center_x_km'] = df['center_x_px'] / PIXELS_PER_KM
        df['center_y_km'] = df['center_y_px'] / PIXELS_PER_KM
        df['width_km'] = df['width_px'] / PIXELS_PER_KM
        df['height_km'] = df['height_px'] / PIXELS_PER_KM
        df['diameter_km'] = (df['width_km'] + df['height_km']) / 2

        # 3. Compute Hazard Metrics (CDI, SWHI, and Safety Scores)
        cell_h_px = img_h / GRID_N
        cell_w_px = img_w / GRID_N
        cell_area_km2 = (cell_h_px / PIXELS_PER_KM) * (cell_w_px / PIXELS_PER_KM)

        count_grid = np.zeros((GRID_N, GRID_N))
        diameter_sum_grid = np.zeros((GRID_N, GRID_N))

        # Build counts and sums
        for idx, row in df.iterrows():
            col_index = int(row['center_x_px'] / cell_w_px)
            row_index = int(row['center_y_px'] / cell_h_px)
            
            col_index = min(col_index, GRID_N - 1)
            row_index = min(row_index, GRID_N - 1)
            
            count_grid[row_index, col_index] += 1
            diameter_sum_grid[row_index, col_index] += row['diameter_km']

        cdi_grid = count_grid / cell_area_km2
        swhi_grid = diameter_sum_grid / cell_area_km2

        # Normalize metrics to construct safety hazard index
        max_cdi = cdi_grid.max() if cdi_grid.max() > 0 else 1.0
        max_swhi = swhi_grid.max() if swhi_grid.max() > 0 else 1.0

        normalized_cdi = cdi_grid / max_cdi
        normalized_swhi = swhi_grid / max_swhi

        safety_score = (normalized_cdi + normalized_swhi) / 2
        safe_mask = safety_score <= SAFETY_THRESHOLD

        # Locate 3 top safest sites
        flat_indices = np.argsort(safety_score, axis=None) # Ascending order (safest first)
        top_cells = []
        for idx in flat_indices:
            r, c = np.unravel_index(idx, (GRID_N, GRID_N))
            top_cells.append((r, c))
            if len(top_cells) == 3:
                break

        # -------------------------------------------------------
        # VISUALIZATIONS & REPORTING
        # -------------------------------------------------------
        st.write("---")
        st.subheader("📊 Landing Site Safety Analytics")
        
        # Display Heatmaps side by side
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7))
        
        # CDI Heatmap
        im_cdi = axes1[0].imshow(cdi_grid, cmap='YlOrRd', interpolation='nearest')
        fig1.colorbar(im_cdi, ax=axes1[0], label='Crater Density (craters/km²)')
        axes1[0].set_title("Crater Density Index (CDI)", fontsize=13, fontweight='bold')
        axes1[0].set_xlabel("Column")
        axes1[0].set_ylabel("Row")
        
        # SWHI Heatmap
        im_swhi = axes1[1].imshow(swhi_grid, cmap='hot', interpolation='nearest')
        fig1.colorbar(im_swhi, ax=axes1[1], label='Sum of crater diameters (km/km²)')
        axes1[1].set_title("Size-Weighted Hazard Index (SWHI)", fontsize=13, fontweight='bold')
        axes1[1].set_xlabel("Column")
        axes1[1].set_ylabel("Row")
        
        st.pyplot(fig1)

        # Plot 2: Final Site Overlay
        st.write("---")
        st.subheader("🎯 Landing Hazard Assessment Overlay")
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 9))

        # Left panel: Detections bounding box
        axes2[0].imshow(img_rgb)
        for idx, row in df.iterrows():
            rect = patches.Rectangle(
                (row['x1'], row['y1']), row['width_px'], row['height_px'],
                linewidth=1.2, edgecolor='cyan', facecolor='none', alpha=0.7
            )
            axes2[0].add_patch(rect)
        axes2[0].set_title(f"Model Detection Map ({len(df)} Craters Identified)", fontsize=14, fontweight='bold')
        axes2[0].axis('off')

        # Right panel: Safety Overlay & Top 3 Site Selections
        axes2[1].imshow(img_rgb)
        # Apply overlay masking
        overlay = np.zeros_like(img_rgb, dtype=np.uint8)
        for r in range(GRID_N):
            for c in range(GRID_N):
                y_start = int(r * cell_h_px)
                y_end = int((r + 1) * cell_h_px)
                x_start = int(c * cell_w_px)
                x_end = int((c + 1) * cell_w_px)
                
                if not safe_mask[r, c]:
                    # Paint Hazardous Zones translucent red
                    overlay[y_start:y_end, x_start:x_end] = [255, 0, 0]
                else:
                    # Paint Safe Zones translucent green
                    overlay[y_start:y_end, x_start:x_end] = [0, 255, 0]

        blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        axes2[1].imshow(blended)

        # Draw grid lines & site priorities
        for i in range(1, GRID_N):
            axes2[1].axhline(y=i*cell_h_px, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
            axes2[1].axvline(x=i*cell_w_px, color='white', linestyle='--', linewidth=0.8, alpha=0.5)

        colors_rank = ['gold', 'silver', 'peru']
        for rank, (r, c) in enumerate(top_cells):
            cx = (c + 0.5) * cell_w_px
            cy = (r + 0.5) * cell_h_px
            circle = patches.Circle((cx, cy), radius=min(cell_w_px, cell_h_px)*0.3, 
                                    linewidth=3, edgecolor=colors_rank[rank], facecolor='none')
            axes2[1].add_patch(circle)
            axes2[1].text(cx, cy, str(rank+1), color='black', fontsize=12, fontweight='bold',
                         ha='center', va='center', bbox=dict(facecolor=colors_rank[rank], edgecolor='none', boxstyle='circle'))

        axes2[1].set_title("Safety Zones (Green=Safe, Red=Hazard) & Top 3 Targets", fontsize=14, fontweight='bold')
        axes2[1].axis('off')
        
        st.pyplot(fig2)

        # Print Top Sites Detailed Info
        st.subheader("🏆 Recommended Landing Zone Details")
        for rank, (r, c) in enumerate(top_cells):
            status = "Highly Safe" if safety_score[r, c] <= SAFETY_THRESHOLD else "Marginal Safety"
            st.metric(
                label=f"Rank {rank+1}: Grid Cell Row {r}, Column {c} ({status})",
                value=f"Safety Score: {safety_score[r, c]:.4f}",
                delta=f"{int(count_grid[r, c])} local craters",
                delta_color="inverse"
            )
    else:
        st.warning("⚠️ No craters detected in the selected image with the current settings.")