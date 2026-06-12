import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_extractor_comparison(video_name, extractor_summaries, gt_summary=None, save_path="./figure"):
    """
    Compare different feature extractors. 
    If gt_summary is None, it only plots the model selections.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Get total frames from the first extractor's output
    n_frames = len(list(extractor_summaries.values())[0])
    num_models = len(extractor_summaries)
    
    fig, ax = plt.subplots(figsize=(15, 2 + num_models))
    
    # 1. Plot Ground Truth ONLY if it exists
    if gt_summary is not None:
        ax.text(-50, 0.5, "Ground Truth", va='center', ha='right', fontsize=10, fontweight='bold')
        for f in range(n_frames):
            if gt_summary[f] == 1:
                # Black outline for ground truth
                ax.add_patch(patches.Rectangle((f, 0), 1, 0.8, edgecolor='black', facecolor='none', lw=1.5))

    # 2. Plot each extractor's binary summary
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#0000FF'] 
    for idx, (ext_name, summary) in enumerate(extractor_summaries.items()):
        y_pos = -idx - 1
        ax.text(-50, y_pos + 0.5, f"Extractor: {ext_name}", va='center', ha='right', fontsize=10)
        
        for f in range(n_frames):
            if summary[f] == 1:
                ax.add_patch(patches.Rectangle((f, y_pos), 1, 0.8, facecolor=colors[idx % len(colors)]))

    ax.set_xlim(0, n_frames)
    # Adjust Y-axis depending on whether GT was plotted
    y_top = 1.5 if gt_summary is not None else 0.5
    ax.set_ylim(-num_models - 1, y_top)
    ax.set_yticks([])
    
    title_suffix = " (vs Ground Truth)" if gt_summary is not None else " (No Ground Truth)"
    plt.title(f"Extractor Comparison for {video_name}{title_suffix}")
    plt.savefig(f"{save_path}/{video_name}_extractors.png", bbox_inches='tight')
    plt.close()


def plot_inference_with_heatmap(video_path, scores, binary_summary, gt_summary=None, save_path="./figure"):
    """
    Plot thumbnails and color-concentrated timeline (Heatmap).
    If gt_summary is provided, a thin black bar indicates human selection below the heatmap.
    """
    os.makedirs(save_path, exist_ok=True)
    video_name = os.path.basename(video_path).split('.')[0]
    n_frames = len(scores)
    
    fig, (ax_img, ax_time) = plt.subplots(2, 1, figsize=(15, 6), gridspec_kw={'height_ratios': [3, 1]})
    
    # --- 1. Extract and Plot Frames (Top row) ---
    cap = cv2.VideoCapture(video_path)
    selected_indices = np.where(binary_summary == 1)[0]
    
    if len(selected_indices) > 0:
        top_frames = sorted(selected_indices, key=lambda x: scores[x], reverse=True)[:4]
        top_frames.sort()
        
        images = []
        for f_idx in top_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if images:
            concat_img = np.concatenate(images, axis=1)
            ax_img.imshow(concat_img)
    
    ax_img.axis('off')
    ax_img.set_title(f"Top Selected Key-shots: {video_name}")
    
    # --- 2. Plot Color Concentration / Heatmap (Bottom row) ---
    cmap = plt.cm.Reds 
    for f in range(n_frames):
        if binary_summary[f] == 1:
            color = cmap(scores[f]) 
            ax_time.add_patch(patches.Rectangle((f, 0), 1, 1, facecolor=color))
            
    # If Ground Truth exists, draw it as a thin black line right below the heatmap
    if gt_summary is not None:
        for f in range(n_frames):
            if gt_summary[f] == 1:
                ax_time.add_patch(patches.Rectangle((f, -0.2), 1, 0.15, facecolor='black'))
        ax_time.set_ylim(-0.3, 1)
    else:
        ax_time.set_ylim(0, 1)

    ax_time.set_xlim(0, n_frames)
    ax_time.set_yticks([])
    ax_time.set_xlabel("Frame Number")
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{video_name}_heatmap.png")
    plt.close()