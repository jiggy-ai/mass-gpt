
from typing import Optional
from pydantic import condecimal
from time import time

import os
from sqlmodel import create_engine, SQLModel, Field, Session, select



# DB Config
db_host = os.environ['HNSUM_POSTGRES_HOST']
user    = os.environ['HNSUM_POSTGRES_USER']
passwd  = os.environ['HNSUM_POSTGRES_PASS']

DBURI = 'postgresql+psycopg2://%s:%s@%s:5432/hnsum' % (user, passwd, db_host)

engine = create_engine(DBURI, pool_pre_ping=True, echo=False)

# Create DB Engine
SQLModel.metadata.create_all(engine)


timestamp = condecimal(max_digits=14, decimal_places=3)


class HackerNewsStory(SQLModel, table=True):
    # partial state of HN Story item. See https://github.com/HackerNews/API#items
    # we dont include "type" since we are only recording type='story' here.    
    id:             int = Field(primary_key=True, description="The item's unique id.")
    by:             str = Field(index=True, description="The username of the item's author.")
    time:           int = Field(index=True, description="Creation date of the item, in Unix Time.")
    title:          str = Field(description="The title of the story, poll or job. HTML.")    
    text: Optional[str] = Field(description="The comment, story or poll text. HTML.")
    url:  Optional[str] = Field(description="The url associated with the Item.")

class StoryText(SQLModel, table=True):
    id:       int = Field(primary_key=True, description="The summary unique id.")
    story_id: int = Field(index=True, description="The story id this text is associated with.")
    mechanism:  str = Field(description="identifies which software mechanism exracted the text from the url")
    crawl_time: timestamp = Field(default_factory=time, description='The epoch timestamp when the url was crawled.')
    html: Optional[str] = Field(description="original html content")
    text:     str = Field(max_length=65535, description="The readable text we managed to extract from the Story Url.")
    
class StorySummary(SQLModel, table=True):
    id:       int = Field(primary_key=True, description="The summary unique id.")
    story_id: int = Field(index=True, description="The story id.")
    model:    str = Field(description="The model used to summarize a story")
    prompt:   str = Field(max_length=65535, description="The prompt used to create the summary.")
    summary:  str = Field(max_length=65535, description="The summary we got back from the model.")
    upvotes:  Optional[int] = Field(default=0, description="The number of upvotes for this summary.")
    votes:    Optional[int] = Field(default=0, description="The total number of votes for this summary.")
    



def stories(story_ids):
    with Session(engine) as session:
        return session.exec(select(HackerNewsStory).where(HackerNewsStory.id.in_(story_ids))).all()

