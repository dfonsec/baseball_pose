import numpy as np
import pandas as pd
import json
import pickle as pk


LANDMARK_NAMES = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

# Horizontal Distance Between Knees
def hor_vert_distance(keypoint_1, keypoint_2, orientation="H"):
    """ Computes horizontal/vertical distance between two points
    
    Args:
        keypoint_1: The keypoint values of a body joint
        
        keypoint_2: The keypoint values of a body joint
        
        orientation: The orientation in which distances are 
                     computed w.r.t 
    
    Returns:
        The absolute value (distance) between these two keypoints
    
    """
    
    if orientation == "H":
        keypoint_1 = keypoint_1[0]
        keypoint_2 = keypoint_2[0]    
    elif orientation == "V":
        keypoint_1 = keypoint_1[1]
        keypoint_2 = keypoint_2[1] 
        
    
    return np.abs(keypoint_1, keypoint_2)


def _compute_angle(a, b, c):
    """Computes angle ABC (in degrees) where B is the vertex
    
    Args:
        a: a vector of coordinates
        
        b: a vector of coordinates (vertex)
        
        c: a vector of coordinates
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


    






