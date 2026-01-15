import numpy as np

def pre_process_landmark(landmark_xy):
    """
    landmark_xy: list of 21 (x, y) tuples in normalized coords.
    Returns: (42,) float32 vector, relative to wrist, normalized.
    """
    pts = np.array(landmark_xy, dtype=np.float32)  # (21,2)

    # relative coordinates: wrist as origin
    base = pts[0].copy()
    pts = pts - base

    # flatten
    flat = pts.reshape(-1)

    # normalize
    max_abs = np.max(np.abs(flat))
    if max_abs < 1e-6:
        max_abs = 1.0
    flat = flat / max_abs

    return flat.astype(np.float32)
