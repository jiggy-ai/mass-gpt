
import hnswlib
from db import engine
from  models import *
from sentence_transformers import SentenceTransformer
from sqlmodel import Session, select
import psutil
import  hn_summary_db

CPU_COUNT = psutil.cpu_count()

## Embedding Config
ST_MODEL_NAME   =  'multi-qa-mpnet-base-dot-v1'
st_model        =  SentenceTransformer(ST_MODEL_NAME)

hnsw_ix = hnswlib.Index(space='cosine', dim=768)
hnsw_ix.load_index('index-multi-qa-mpnet-base-dot-v1-0.hnsf', max_elements=20000)
hnsw_ix.set_ef(1000)

STORY_SOURCES = [EmbeddingSource.hn_story_summary, EmbeddingSource.hn_story_title]


def search(query : str):
    query = query.rstrip()
    print(query)
    vector =  st_model.encode(query)

    ids, distances = hnsw_ix.knn_query([vector], k=10)
    ids = [int(i) for i in ids[0]]
    distances = [float(i) for i in distances[0]]
    with Session(engine) as session:
        ann_embeddings = session.exec(select(Embedding).where(Embedding.id.in_(ids))).all()
        hn_story_ids = [e.source_id for e in ann_embeddings if e.source in STORY_SOURCES]
        stories = hn_summary_db.stories(hn_story_ids)

    vid_to_distance = {vid : distance for vid,distance in zip(ids, distances)}
    sid_to_vid = {emb.source_id : emb.id for emb in ann_embeddings}

    def distance(story):
        return vid_to_distance[sid_to_vid[story.id]]
    
    stories.sort(key=distance)
    
    for story in stories:
        print(f"{distance(story):.2f}  {story.title}")


if __name__ == "__main__":
    search("SBF fraud")
    while(True):
        search(input("Search: "))








