import pickle as pkl
import psycopg2
from psycopg2 import sql
from collections import defaultdict
from collections import Counter


with open("/Users/danielfonseca/repos/baseball_pose/pipeline/professional_embeddings.pkl", "rb") as file: 
    professional_embeddings = pkl.load(file)
    
# must match the order of features in your PlayerSwingEmbedding
FEATURE_TABLES = [
    'knee_left',
    'knee_right',
    'elbow_left',
    'elbow_right',
    'hip_dist',
    'shoulder_angle',
    'back_arm_body',
    'groin_knee_left',
    'groin_knee_right'
]


def insert_player_embeddings(professional_embeddings):

    conn = psycopg2.connect(
    dbname="angle_embeddings",
    user="postgres",
    password="postgres"
                            )

    cur = conn.cursor()
    for player_name, curves in professional_embeddings.items():
        for table_name, vector in zip(FEATURE_TABLES, curves):
            # convert to Python list for psycopg2
            vec_list = vector.tolist()
            # build a query like:
            #   INSERT INTO knee_left (player_name, embed) VALUES (%s, %s)
            query = sql.SQL("""
                INSERT INTO {tbl} (player_id, embed)
                VALUES (%s, %s::vector)
            """).format(
                tbl=sql.Identifier(table_name)
            )
            cur.execute(query, [player_name, vec_list])

    # commit once at the end
    conn.commit()
    cur.close()
    conn.close()

    return
    
    
    
def classify_amateur(curves, conn, top_k=1):
    cur = conn.cursor()
    votes = []

    for table_name, vec in zip(FEATURE_TABLES, curves):
        # 1) build the query that casts to vector
        query = sql.SQL("""
            SELECT player_id
            FROM {tbl}
            ORDER BY embed <-> %s::vector
            LIMIT %s
        """).format(tbl=sql.Identifier(table_name))

        # 2) convert your numpy array to a pgvector literal string
        vec_str = '[' + ','.join(f'{float(x):.6f}' for x in vec) + ']'

        # 3) execute with (vector_literal, k)
        cur.execute(query, (vec_str, top_k))
        for (player_name,) in cur.fetchall():
            votes.append(player_name)

    cur.close()
    # pick the player with the most “votes”
    return Counter(votes).most_common(1)[0][0]


def main():
    
    with open("/Users/danielfonseca/repos/baseball_pose/pipeline/amateur_embeddings.pkl", "rb") as file: 
        amateur_embeddings = pkl.load(file)
        
    conn = psycopg2.connect(
        dbname="angle_embeddings",
        user="postgres",
        password="postgres"
    )
    
    for am_name, curves in amateur_embeddings.items():
        pred = classify_amateur(curves, conn)
        print(f"{am_name}  → predicted pro match: {pred}")

    conn.close()
        
    return

if __name__ == '__main__':
    main()
    









    
    


