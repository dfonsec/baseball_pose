from torchreid.utils import FeatureExtractor
import cv2
import numpy as np
import os
import torch
import pgvector
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector



def main():
    
    
    extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='/Users/danielfonseca/repos/baseball_pose/deep-person-reid/torchreid/models/osnet_x1_0_imagenet.pth',
    device='cpu'
)

    # vids = [
    #     ["Betts", "/Users/danielfonseca/repos/baseball_pose/vids/betts_cropped.mov"],
    #     ["Bobbdy","/Users/danielfonseca/repos/baseball_pose/vids/bobbdy_cropped.mov"],
    #     ["Tatis", "/Users/danielfonseca/repos/baseball_pose/vids/tatis_cropped.mov"]
    # ]
    # vids_numpy = []
    
    
    # for name, vid_path in vids:
    #     frames = []
    #     cap = cv2.VideoCapture(vid_path)
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frames.append(frame)
    
    #     cap.release()
    #     vids_numpy.append([name, frames])

    # embeddings = []
    # for name, frames_np in vids_numpy:
    #     features = extractor(frames_np)
    #     video_embedding = features.mean(dim=0)
    #     video_embedding = video_embedding.detach().cpu().numpy()

    #     embeddings.append([name, video_embedding])
        
    # conn = psycopg2.connect("dbname=player_swings user=postgres password=postgres")
    # cur = conn.cursor()

    # for player_name, embedding_np in embeddings:
    #     embedding_list = embedding_np.tolist()
    #     cur.execute(
    #         "INSERT INTO items (name, embedding) VALUES (%s, %s)",
    #         (player_name, embedding_list)
    #     )

        
    # conn.commit()
    # cur.close()
    # conn.close()
        
    query_path = '/Users/danielfonseca/repos/baseball_pose/vids/tatis_2.mp4'
    
    frames = []
    cap = cv2.VideoCapture(query_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    features_query = extractor(frames)
    query_embedding = features_query.mean(dim=0)
    query_embedding = query_embedding.detach().cpu().numpy()
    
    conn = psycopg2.connect("dbname=player_swings user=postgres password=postgres")
    cur = conn.cursor()
    
    # We are expecting to see Tatis to be first (ideally :( )
    
    new_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]" 
    
    cur.execute("""
    SELECT name, embedding, 
       embedding <=> %s AS similarity_score  -- Use <-> for Euclidean distance
    FROM items
    ORDER BY similarity_score ASC  -- Ascending because lower Euclidean distance means more similar
    LIMIT 5;
    """, (new_embedding_str,))
    
    results = cur.fetchall()
    
    for result in results:
        player_name, embedding, similarity_score = result
        print(f"Player: {player_name}, Similarity Score: {similarity_score}")
        
    cur.close()
    conn.close()
    
    return



if __name__ == '__main__':
    main()