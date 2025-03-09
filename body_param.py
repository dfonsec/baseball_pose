import numpy as np
import json
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


class ParamPose:
    def __init__(self, kps_json_path):
        
        with open(kps_json_path, 'r') as kps:
            json_kps = json.load(kps)
            
        self.raw_kps = json_kps
        self.normalized_kps = self._normalize_kps()
        self.parameters = self._extract_parameters()
        
    
    def _normalize_kps(self):
        
        for frame_data in self.raw_kps:
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
        
        return self.raw_kps
    
    def _extract_parameters(self):
        params = {}
        

        params["left_elbow_angle"] = self._extract_elbow_angle()
        
        return params
    
    
    def _compute_angle(self, a, b, c):
        
        ba = np.array([a['x'] - b['x'], a['y'] - b['y']])
        bc = np.array([c['x'] - b['x'], c['y'] - b['y']])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _extract_elbow_angle(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            
            if frame_data["frame"] in seen_frames:
                continue
            else:
                angle = self._compute_angle(frame_data["keypoints"]["LEFT_SHOULDER"], 
                                            frame_data["keypoints"]["LEFT_ELBOW"],
                                            frame_data["keypoints"]["LEFT_WRIST"])
                       
                seen_frames.add(frame_data["frame"])
                res.append(angle)
            
        return res

def resample_time_series(data, num_samples=155):
    
        original_length = len(data)
        # Normalized time points for the original data
        original_times = np.linspace(0, 1, original_length)
        # Desired normalized time points for the resampled data
        resample_times = np.linspace(0, 1, num_samples)
        # Interpolate to resample the data
        resampled_data = np.interp(resample_times, original_times, data)
        
        return resampled_data
    
    
    

        
        
           
           
def main():
    acuna_1 = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/acuna_1.json")
    acuna_2 = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/acuna_2.json")
    tatis = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/tatis.json")
    
    
    acuna_norm1 = resample_time_series(acuna_1.parameters["left_elbow_angle"]) # normal
    acuna_norm2 = resample_time_series(acuna_2.parameters["left_elbow_angle"]) #slowmo 
    tatis_norm = resample_time_series(tatis.parameters["left_elbow_angle"]) # slowmo
    distance = dtw.distance(tatis_norm, acuna_norm2)
    print(distance)

    return


if __name__ == "__main__":
    main()

    
                
    
        
    
    