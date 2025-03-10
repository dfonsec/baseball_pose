import numpy as np
import json
from dtaidistance.subsequence.dtw import subsequence_alignment
from fastdtw import fastdtw
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns

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

#TODO: Develop other metrics such as load

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
        

        params["left_elbow_angle"] = self._extract_elbow_angle_left()
        params["right_elbow_angle"] = self._extract_elbow_angle_right()
        params["left_knee_angle"] = self._extract_knee_angle_left()
        params["right_knee_angle"] = self._extract_knee_angle_right()
        params["right_elbow_height"] = self._extract_elbow_height_right()
        params["left_elbow_height"] = self._extract_elbow_height_left()
        params["knees_distance"] = self._extract_knee_distance()
        # define more parameters
        
       
        
        return params
    
    
    def _compute_knee_distance(self, left_knee, right_knee):
        dx = left_knee["x"] - right_knee["x"]
        dy = left_knee["y"] - right_knee["y"]
        dz = left_knee["z"] - right_knee["z"]
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
        
    def _compute_angle(self, a, b, c):
        
        ba = np.array([a['x'] - b['x'], a['y'] - b['y']])
        bc = np.array([c['x'] - b['x'], c['y'] - b['y']])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _extract_elbow_angle_left(self):
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

    def _extract_elbow_angle_right(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            
            if frame_data["frame"] in seen_frames:
                continue
            else:
                angle = self._compute_angle(frame_data["keypoints"]["RIGHT_SHOULDER"], 
                                            frame_data["keypoints"]["RIGHT_ELBOW"],
                                            frame_data["keypoints"]["RIGHT_WRIST"])
                       
                seen_frames.add(frame_data["frame"])
                res.append(angle)
            
        return res

    def _extract_knee_angle_left(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            
            if frame_data["frame"] in seen_frames:
                continue
            else:
                angle = self._compute_angle(frame_data["keypoints"]["LEFT_HIP"], 
                                            frame_data["keypoints"]["LEFT_KNEE"],
                                            frame_data["keypoints"]["LEFT_ANKLE"])
                seen_frames.add(frame_data["frame"])
                res.append(angle)
        
        return res
    
    def _extract_knee_angle_right(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            
            if frame_data["frame"] in seen_frames:
                continue
            else:
                angle = self._compute_angle(frame_data["keypoints"]["RIGHT_HIP"], 
                                            frame_data["keypoints"]["RIGHT_KNEE"],
                                            frame_data["keypoints"]["RIGHT_ANKLE"])
                seen_frames.add(frame_data["frame"])
                res.append(angle)
        
        return res
    

    def _extract_elbow_height_left(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            if frame_data["frame"] in seen_frames:
                continue
            else:
                left_elbow_height = frame_data["keypoints"]["LEFT_ELBOW"]["y"]
                seen_frames.add(frame_data["frame"])
                res.append(left_elbow_height)
        
        return res
    
    def _extract_elbow_height_right(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            if frame_data["frame"] in seen_frames:
                continue
            else:
                right_elbow_height = frame_data["keypoints"]["RIGHT_ELBOW"]["y"]
                seen_frames.add(frame_data["frame"])
                res.append(right_elbow_height)
        
        return res
    
    def _extract_knee_distance(self):
        res = []
        seen_frames = set()
        for frame_data in self.normalized_kps:
            if frame_data["frame"] in self.normalized_kps:
                continue
            else:
                distance = self._compute_knee_distance(frame_data["keypoints"]["LEFT_KNEE"],
                                                       frame_data["keypoints"]["RIGHT_KNEE"]
                                                       )
                seen_frames.add(frame_data["frame"])
                res.append(distance)
        
        return res
            
def resample_time_series(data, num_samples=200):
    
        original_length = len(data)
        
        # Normalized time points for the original data
        original_times = np.linspace(0, 1, original_length)
        # Desired normalized time points for the resampled data
        resample_times = np.linspace(0, 1, num_samples)
        # Interpolate to resample the data
        
        
        interpolator = interp1d(original_times, data, kind='linear')
        
        return interpolator(resample_times)
    
def dtw_compare(label_data, input_data):
    keys = [
        "left_elbow_angle", "right_elbow_angle",
        "left_knee_angle", "right_knee_angle",
        "right_elbow_height", "left_elbow_height",
        "knees_distance"
    ]

    scores = {}
    
    # Compute DTW distance and score for each parameter
    for key in keys:
        # Each key's value is a list representing the time-series across frames
        label_series = resample_time_series(label_data.get(key, []))
        input_series = resample_time_series(input_data.get(key, []))
        
        # Compute the DTW distance between the two time-series
        if len(label_series) <= len(input_series):
            query_series, reference_series = label_series, input_series
        else:
            query_series, reference_series = input_series, label_series
    
        alignment = subsequence_alignment(query_series, reference_series)
        match = alignment.best_match()
        startidx, endidx = match.segment
       

        matched_reference_series = reference_series[startidx:endidx]

        distance, _ = fastdtw(query_series, matched_reference_series)

        scores[key] = distance * 100
    
    # Calculate an overall score as the average of the individual scores
    scores["Total_Score"] = np.mean(list(scores.values()))
    
    return scores
    

        
           
def main():
    acuna_1 = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/acuna_1.json")
    acuna_2 = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/acuna_2.json")
    betts = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/betts.json")
    betts_2 = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/betts_2.json")
    tatis = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/tatis.json")
    abreu = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/abreu.json")
    guerrero = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/guerrero.json")
    rodriguez = ParamPose("/Users/danielfonseca/repos/baseball_pose/pose_jsons/rodriguez.json")
    
    # Acuna and Acuna
    scores_acuna = dtw_compare(acuna_1.parameters, acuna_2.parameters)
    
    # Acunas and Betts
    scores_acuna1_betts = dtw_compare(acuna_1.parameters, betts.parameters)
    scores_acuna2_betts = dtw_compare(acuna_2.parameters, betts.parameters)
    
    # Acunas and Tatis
    scores_acuna1_tatis = dtw_compare(acuna_1.parameters, tatis.parameters)
    scores_acuna2_tatis = dtw_compare(acuna_2.parameters, tatis.parameters)
    
    # Acunas and Abreu
    scores_acuna1_abreu = dtw_compare(acuna_1.parameters, abreu.parameters)
    scores_acuna2_abreu = dtw_compare(acuna_2.parameters, abreu.parameters)
    
    # Acunas and Guerrero
    scores_acuna1_guerrero = dtw_compare(acuna_1.parameters, guerrero.parameters)
    scores_acuna2_guerrero = dtw_compare(acuna_2.parameters, guerrero.parameters)
    
    # Acunas and Rodriguez
    scores_acuna1_rodriguez = dtw_compare(acuna_1.parameters, rodriguez.parameters)
    scores_acuna2_rodriguez = dtw_compare(acuna_2.parameters, rodriguez.parameters)
    
    #Betts and Betts
    scores_betts_betts_2 = dtw_compare(betts.parameters, betts_2.parameters)
    
    # Betts and Tatis 
    scores_betts_tatis = dtw_compare(tatis.parameters, betts.parameters)
    scores_betts_2_tatis = dtw_compare(tatis.parameters, betts_2.parameters)
    
    # Betts and Abreu
    scores_betts_abreu = dtw_compare(abreu.parameters, betts.parameters)
    scores_betts_2_abreu = dtw_compare(abreu.parameters, betts_2.parameters)
    
    #Betts and Guerrero
    scores_betts_guerrero = dtw_compare(guerrero.parameters, betts.parameters)
    scores_betts_2_guerrero = dtw_compare(guerrero.parameters, betts_2.parameters)
    
    # Betts and Rodriguez
    scores_betts_rodriguez = dtw_compare(rodriguez.parameters, betts.parameters)
    scores_betts_2_rodriguez = dtw_compare(rodriguez.parameters, betts_2.parameters)

    scores = [
    scores_acuna["Total_Score"],
    scores_acuna1_betts["Total_Score"],
    scores_acuna2_betts["Total_Score"],
    scores_acuna1_tatis["Total_Score"],
    scores_acuna2_tatis["Total_Score"],
    scores_acuna1_abreu["Total_Score"],
    scores_acuna2_abreu["Total_Score"],
    scores_acuna1_guerrero["Total_Score"],
    scores_acuna2_guerrero["Total_Score"],
    scores_acuna1_rodriguez["Total_Score"],
    scores_acuna2_rodriguez["Total_Score"],
    scores_betts_betts_2["Total_Score"],
    scores_betts_tatis["Total_Score"],
    scores_betts_2_tatis["Total_Score"],
    scores_betts_abreu["Total_Score"],
    scores_betts_2_abreu["Total_Score"],
    scores_betts_guerrero["Total_Score"],
    scores_betts_2_guerrero["Total_Score"],
    scores_betts_rodriguez["Total_Score"],
    scores_betts_2_rodriguez["Total_Score"]
]
    
    labels = [
    "Acuna1 vs Acuna2",
    "Acuna1 vs Betts",
    "Acuna2 vs Betts",
    "Acuna1 vs Tatis",
    "Acuna2 vs Tatis",
    "Acuna1 vs Abreu",
    "Acuna2 vs Abreu",
    "Acuna1 vs Guerrero",
    "Acuna2 vs Guerrero",
    "Acuna1 vs Rodriguez",
    "Acuna2 vs Rodriguez",
    "Betts vs Betts2",
    "Betts vs Tatis",
    "Betts2 vs Tatis",
    "Betts vs Abreu",
    "Betts2 vs Abreu",
    "Betts vs Guerrero",
    "Betts2 vs Guerrero",
    "Betts vs Rodriguez",
    "Betts2 vs Rodriguez"
]
    
    plt.figure(figsize=(20, 6))

    for i, (score, label) in enumerate(zip(scores, labels)):
        color = np.random.rand(3)
        plt.scatter(i, score, color=color)
        plt.annotate(label, (i, score), textcoords="offset points", xytext=(0, 10), ha="center")


    plt.xticks(range(len(scores)), labels, rotation=45, ha="right")
    plt.ylabel("DTW Total Score")
    plt.title("DTW Comparison Scores for Player Poses (Lower is Better)")
    plt.tight_layout()
    plt.savefig("Test_2.png")
    
    return


if __name__ == "__main__":
    main()
    
    
    
    
    

    
                
    
        
    
    