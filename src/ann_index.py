import hnswlib
import psutil

from s3 import  bucket

from loguru import logger
from sqlmodel import Session, select
from db import engine
from  models import *

CPU_COUNT = psutil.cpu_count()
ST_MODEL_NAME   =  'multi-qa-mpnet-base-dot-v1'


hnsw_index = hnswlib.Index(space='cosine', dim=768)
hnsw_index.set_num_threads(int(CPU_COUNT/2))

hnsw_index.init_index(max_elements    = 20000,
                      ef_construction = 100,
                      M               = 16)

count = 0
with Session(engine) as session:
    for embedding in session.exec(select(Embedding).where(Embedding.collection == ST_MODEL_NAME)):
        if embedding.vector is None:
            session.delete(embedding)
            session.commit()
            continue        
        hnsw_index.add_items([embedding.vector], [embedding.id])
        count += 1
        print(embedding.id)

filename = "index-%s-%d.hnsf" % (ST_MODEL_NAME, count)
hnsw_index.save_index(filename)

objkey = f"massgpt/ann-index/{ST_MODEL_NAME}-{count}.hnsw"
bucket.upload_file(filename, objkey)

with Session(engine) as session:
    ix = HnswIndex(collection = ST_MODEL_NAME,
                   count      = count,
                   objkey     = objkey)
    session.add(ix)
    session.commit()
