import cv2
import numpy as np

def compile_summary_video(input_video_path, binary_summary, output_video_path):
    """Slices the original video frames based on the binary mask and creates a summary video."""
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Check if this frame index is selected inside our summary length
        if frame_idx < len(binary_summary) and binary_summary[frame_idx]:
            out.write(frame)
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Video summary written successfully to: {output_video_path}")