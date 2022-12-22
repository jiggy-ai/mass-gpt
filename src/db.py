# database engine
import os
from sqlmodel import create_engine, SQLModel



# DB Config
db_host = os.environ['MASSGPT_POSTGRES_HOST']
user = os.environ['MASSGPT_POSTGRES_USER']
passwd = os.environ['MASSGPT_POSTGRES_PASS']

DBURI = 'postgresql+psycopg2://%s:%s@%s:5432/massgpt' % (user, passwd, db_host)

engine = create_engine(DBURI, pool_pre_ping=True, echo=False)

if __name__ == "__main__":
    from models import *
    SQLModel.metadata.create_all(engine)
    print("create_all complete")
