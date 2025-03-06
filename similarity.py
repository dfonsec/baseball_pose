import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dtaidistance import dtw

ordered_keypoints = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


def l2_normalize(jsonfile):
    
    with open(jsonfile, 'r') as kps:
        json_kps = json.load(kps)
        
        for frame_data in json_kps:
            keypoints = frame_data['keypoints']
            box = frame_data['box']
            temp_x = np.abs(box[0] - box[2]) / 2
            temp_y = np.abs(box[1] - box[3]) / 2
            
            if temp_x <= temp_y:
                if box[0] <= box[2]:
                    sub_x = box[0] - (temp_y - temp_x)
                else:
                    sub_x = box[2] - (temp_y - temp_x)
                
                if box[1] <= box[3]:
                    sub_y = box[1]
                else:
                    sub_y = box[3]
            else:
                if box[1] <= box[3]:
                    sub_y = box[1] - (temp_x - temp_y)
                else:
                    sub_y = box[3] - (temp_x - temp_y)
                
                if box[0] <= box[2]:
                    sub_x = box[0]
                else:
                    sub_x = box[2]
            
            temp = []
            for key in ordered_keypoints:
                temp.append(keypoints[key]['x'] - sub_x)
                temp.append(keypoints[key]['y'] - sub_y)
            
            norm = np.linalg.norm(temp)
            
            for key in ordered_keypoints:
                keypoints[key]['x'] = (keypoints[key]['x'] - sub_x) / norm
                keypoints[key]['y'] = (keypoints[key]['y'] - sub_y) / norm
                frame_data['keypoints'] = keypoints 
                
    with open(jsonfile.replace('.json', '_l2norm.json'), 'w') as f:
        json.dump(json_kps, f)
        print('Write l2_norm keypoints')

    


def main():
    l2_normalize("/Users/danielfonseca/repos/baseball_pose/video_keypoints.json")
    
    return



if __name__ == "__main__":
    main()
                
                
                