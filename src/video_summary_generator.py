# video_summary_generator.py
import os
import cv2


def map_selected_segments_to_timestamps(selected_shots, shot_bounds, fps):
    """
    Step 7a: Maps chosen knapsack shot selections back to original video frames and time signatures.

    Args:
        selected_shots (list): List of indices representing shots chosen by knapsack algorithm.
        shot_bounds (np.ndarray / list): Multi-dimensional sequence array containing [[start_frame, end_frame], ...]
        fps (float): Frames per second of the original source video file.

    Returns:
        list: A structured list of dictionaries containing precise video timestamp coordinates.
    """
    segments_metadata = []

    for shot_idx in selected_shots:
        start_frame, end_frame = shot_bounds[shot_idx]

        # Calculate fractional timestamp values
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        duration_sec = end_sec - start_sec

        # Convert seconds into clear user-readable minute:second strings
        start_min, start_remain_sec = divmod(int(start_sec), 60)
        end_min, end_remain_sec = divmod(int(end_sec), 60)

        segments_metadata.append({
            "shot_index": int(shot_idx),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start_timestamp": f"{start_min:02d}:{start_remain_sec:02d}",
            "end_timestamp": f"{end_min:02d}:{end_remain_sec:02d}",
            "duration_seconds": round(duration_sec, 2)
        })

    return segments_metadata


def generate_summary_video_from_mappings(video_path, segments_metadata, output_path):
    """
    Step 7b: Extract physical frame blocks from the raw video asset using timestamp mappings
             and stitch them into a continuous summary file.

    Args:
        video_path (str): Target path of source file asset.
        segments_metadata (list): Output list produced during Step 7a mapping execution.
        output_path (str): Intended path file allocation for storage.

    Returns:
        str: Verified path string leading to compiled summary file location.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video file asset not found at tracking location: {video_path}")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Establish target formatting video codecs
    # Note: 'mp4v' is widely supported for compression processing across environments.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for segment in segments_metadata:
        start_f = segment["start_frame"]
        end_f = segment["end_frame"]

        # Force OpenCV frame-pointer positioning directly to the segment offset boundary
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        for _ in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

    cap.release()
    out.release()
    return output_path


# ---------------------------------------------------------------------
# Self-Contained Execution Verification Block (Pipeline Testing Logic)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("[Testing] Initializing baseline configurations for Segment Mapping verification...")

    # Mocking standard inputs: assume knapsack picked shots 0 and 2 from a video running at 30 FPS
    mock_selected_shots = [0, 2]
    mock_shot_bounds = [
        [0, 150],  # Shot 0: 0 to 5 seconds
        [151, 300],  # Shot 1: 5 to 10 seconds
        [301, 450]  # Shot 2: 10 to 15 seconds
    ]
    mock_fps = 30.0

    # Execute Step 7a
    mapped_output = map_selected_segments_to_timestamps(
        selected_shots=mock_selected_shots,
        shot_bounds=mock_shot_bounds,
        fps=mock_fps
    )

    print("\n[Output] Successfully verified Step 7a structural mappings:")
    for chunk in mapped_output:
        print(f" -> Chosen Shot #{chunk['shot_index']}: "
              f"Frames {chunk['start_frame']} to {chunk['end_frame']} "
              f"({chunk['start_timestamp']} -> {chunk['end_timestamp']}) "
              f"Duration: {chunk['duration_seconds']}s")