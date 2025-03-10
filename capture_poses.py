import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plot
import json 



    
def pose(video_path, output_video_path, output_json):
   mp_pose = mp.solutions.pose
   pose = mp_pose.Pose(static_image_mode=False,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
   
   
   
   cap = cv2.VideoCapture(video_path)
   
   frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   
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
           landmarks = results.pose_landmarks.landmark
           xs = [int(landmark.x * frame_width) for landmark in landmarks]
           ys = [int(landmark.y * frame_height) for landmark in landmarks]
           zs = [int(landmark.z * frame_width) for landmark in landmarks]

           
           min_x, max_x = min(xs), max(xs)
           min_y, max_y = min(ys), max(ys)
           min_z, max_z = min(zs), max(zs)
           cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
           
           frame_keypoints = {
               'frame': frame_idx,
               'keypoints': {},
               'box': [min_x, min_y, min_z, max_x, max_y, max_z]
           }
           
           for idx, landmark in enumerate(results.pose_landmarks.landmark):
                frame_keypoints['keypoints'][mp_pose.PoseLandmark(idx).name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility,
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
    
   with open(output_json, 'w') as f:
        json.dump(video_keypoints, f, indent=4)

   print(f'Keypoints data saved to {output_json}')
   print(f'Output video with keypoints saved to {output_video_path}')
   
    

def main():
    
    names = ['abreu', 'guerrero', 'rodriguez']
    for name in names:
        pose(f"/Users/danielfonseca/repos/baseball_pose/vids/{name}.mp4", 
             f"/Users/danielfonseca/repos/baseball_pose/vids/landmarks/{name}_landmarks.mp4",
             f"/Users/danielfonseca/repos/baseball_pose/pose_jsons/{name}.json"
             )
   
    return



if __name__ == '__main__':
    main()