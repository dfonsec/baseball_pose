from torchreid.utils import FeatureExtractor
import cv2
import numpy as np
import os
import torch
import pgvector
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import pickle



def generateEmbeddings(path_list):
    vids_numpy = []
    
    extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='/Users/danielfonseca/repos/baseball_pose/deep-person-reid/torchreid/models/osnet_x1_0_imagenet.pth',
    device='cpu'
)

    for name, vid_path in path_list:
        print(f"Generating Embedding For {name}")
        frames = []
        cap = cv2.VideoCapture(vid_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    
        cap.release()
        vids_numpy.append([name, frames])

    embeddings = []
    for name, frames_np in vids_numpy:
        with torch.no_grad():
            features = extractor(frames_np)
        video_embedding = features.mean(dim=0)
        video_embedding = video_embedding.detach().cpu().numpy()

        embeddings.append([name, video_embedding])
    
    print("Embeddings Successfully Generated")
    return embeddings

def insertEmbeddingsToDB(embeddings):
    
    conn = psycopg2.connect("dbname=player_swings user=postgres password=postgres")
    cur = conn.cursor()

    for player_name, embedding_np in embeddings:
        embedding_list = embedding_np.tolist()
        cur.execute(
            "INSERT INTO items (name, embedding) VALUES (%s, %s)",
            (player_name, embedding_list)
        )

        
    conn.commit()
    cur.close()
    conn.close()
    print("Successfully inserted embeddings into DB")
    return 

def queryDB(embedding):
    
    print(f"QUERYING DATABASE FOR {embedding[0]}")
    conn = psycopg2.connect("dbname=player_swings user=postgres password=postgres")
    cur = conn.cursor()
    
    # We are expecting to see Tatis to be first (ideally :( )
    query_embedding = embedding[1]
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
    print("Query Finished Processing")
    
    return 

def main():
    
    vid_paths = [["judge", "/Users/danielfonseca/repos/baseball_pose/vids/judge.mov"],
                 ["longoria", "/Users/danielfonseca/repos/baseball_pose/vids/longoria.mov"],
                 ["trout", "/Users/danielfonseca/repos/baseball_pose/vids/trout.mov"],
                 ["bichette", "/Users/danielfonseca/repos/baseball_pose/vids/bichette.mov"],
                 ["cabrera", "/Users/danielfonseca/repos/baseball_pose/vids/cabrera.mov"],
                 ["judge_query", "/Users/danielfonseca/repos/baseball_pose/vids/judge_query.mov"],
                 ["longoria_query", "/Users/danielfonseca/repos/baseball_pose/vids/longoria_query.mov"],
                 ["trout_query", "/Users/danielfonseca/repos/baseball_pose/vids/trout_query.mov"],
                 ["bichette_query", "/Users/danielfonseca/repos/baseball_pose/vids/bichette_query.mov"],
                 ["cabrera_query", "/Users/danielfonseca/repos/baseball_pose/vids/cabrera_query.mov"]]

    embeddings = generateEmbeddings(vid_paths)
    
    embeddings_db = embeddings[:5]
    embeddings_query = embeddings[:5]
    insertEmbeddingsToDB(embeddings_db)
    
    
    with open ('query_embeddings_2.pkl', 'wb') as file:
        pickle.dump(embeddings_query, file)
        
    print("Main Function Done")
    
    return



if __name__ == '__main__':
    main()