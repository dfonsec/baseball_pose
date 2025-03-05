import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plot
import json 


    

def main():
    
   mp_pose = mp.solutions.pose
   pose = mp_pose.Pose(static_image_mode=False,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
   
   video_path = '/Users/danielfonseca/repos/baseball_pose/vids/acuna_1.mp4'
   
   cap = cv2.VideoCapture(video_path)
   
   frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   
   output_video_path = '/Users/danielfonseca/repos/baseball_pose/vids/acuna_1_landmarks.mp4'
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
   
   video_keypoints = []
   
   mp_drawing = mp.solutions.drawing_utils
   mp_drawing_styles = mp.solutions.drawing_styles
   
   frame_idx = 0
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
       
       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       results = pose.process(frame_rgb)
       
       if results.pose_landmarks:
           frame_keypoints = {
               'frame': frame_idx,
               'keypoints': {}
           }
           
           for idx, landmark in enumerate(results.pose_landmarks.landmark):
                frame_keypoints['keypoints'][mp_pose.PoseLandmark(idx).name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
                video_keypoints.append(frame_keypoints)
            
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                ) 
          
       out.write(frame)
       frame_idx += 1
   cap.release()
   out.release()
    
   output_json = "video_keypoints.json"
   with open(output_json, 'w') as f:
        json.dump(video_keypoints, f, indent=4)

   print(f'Keypoints data saved to {output_json}')
   print(f'Output video with keypoints saved to {output_video_path}')

        
   return



if __name__ == '__main__':
    main()