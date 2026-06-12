import torch
import numpy as np
import os
from models.classes.mars import MARs 
from kts.cpd_auto import cpd_auto # Import Kernel Temporal Segmentation
from generate_summary import generate_summary
from generate_video import compile_summary_video
from visualize import (
    plot_extractor_comparison, 
    plot_inference_with_heatmap, 
    # plot_score_vs_groundtruth, 
    # plot_summary_budget_distribution
)

def main(input_dim: int): ## Input ##############
    config = get_config()
    os.makedirs(config.output_dir, exist_ok=True)
    video_name = os.path.basename(config.input_video).split('.')[0]
    
    # 1. Pipeline Layer 1: Feature Extraction
    print("Extracting video features...")
    features, n_frames, picks = extract_features_from_raw_video(config.input_video, device=config.device)
    
    # 2. Pipeline Layer 2: Model Prediction Passing
    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(config.device) # Add batch dim
    model = MARs(input_dim)
    # Load your trained model weights (e.g., from split 1)
    model.load_state_dict(torch.load(f'./weights/{config.dataset_name}/split1.pt', map_location=config.device))
    model.eval()
    
    with torch.no_grad():
        output = model(feature_tensor)
    scores = output.squeeze().cpu().numpy().tolist()
    
    # 3. Pipeline Layer 3: Calculate Change Points via KTS & Apply Knapsack
    # Calculate video self-similarity matrix for segment boundaries
    X = np.array(features)
    K = np.dot(X, X.T)
    n_backbone_frames = len(features)
    change_points, _ = cpd_auto(K, n_backbone_frames - 1, 1)
    
    # Map back change points from downsampled frame sequence to original frame counts
    change_points_original = []
    for cp in change_points:
        start_orig = picks[cp[0]]
        end_orig = picks[min(cp[1], len(picks)-1)]
        change_points_original.append([start_orig, end_orig])
    change_points_original = np.array(change_points_original)
    
    print("Optimizing frame selection via Knapsack...")
    summary = generate_summary([change_points_original], [scores], [n_frames], [picks])[0]
    
    # 4. Pipeline Layer 4: Render Summary Video
    output_video_path = os.path.join(config.output_dir, f"{video_name}_summary.mp4")
    compile_summary_video(config.input_video, summary, output_video_path)
    
    # 5. Pipeline Layer 5: Execute Visualizations Matrix
    print("Generating diagnostic visualizations...")
    # Map short downsampled score array to a full timeline for graphing
    full_frame_scores = np.zeros(n_frames)
    for idx, pick_pos in enumerate(picks):
        if idx + 1 < len(picks):
            full_frame_scores[pick_pos:picks[idx+1]] = scores[idx]
        else:
            full_frame_scores[pick_pos:] = scores[idx]
            
    plot_inference_with_heatmap(config.input_video, full_frame_scores, summary, gt_summary=None, save_path=config.output_dir)
    # plot_score_vs_groundtruth(video_name, full_frame_scores, user_scores=None, save_path=config.output_dir)
    # plot_summary_budget_distribution(video_name, change_points_original, summary, save_path=config.output_dir)
    
    print("Inference completely finished!")

if __name__ == "__main__":
    main()