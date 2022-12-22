
# MassGPT sqlmodel models
# Copyright (C) 2022 William S. Kish

from loguru import logger
from typing import Optional, List
from array import array
import enum
from sqlmodel import Field, SQLModel, Column, ARRAY, Float, Enum
from pydantic import  BaseModel, ValidationError, validator
from pydantic import condecimal
from time import time
import telegram

timestamp = condecimal(max_digits=14, decimal_places=3)  # unix epoch timestamp decimal to millisecond precision


class UserStatus(str, enum.Enum):
    enabled  =  "enabled"
    disabled =  "disabled"
    

class User(SQLModel, table=True):
    id:                      int                  = Field(primary_key=True, description='Unique user ID')
    username:                str                  = Field(index=True, description="Username")
    created_at:              timestamp            = Field(default_factory=time, description='The epoch timestamp when the Evaluation was created.')
    first_name:              Optional[str]        = Field(description="User's first name")
    last_name:               Optional[str]        = Field(description="User's last name")
    auth0_id:                Optional[str]        = Field(index=True, description='Auth0 user_id')    
    telegram_id:             Optional[int]        = Field(index=True, description='Telegram User ID')
    telegram_is_bot:         Optional[bool]       = Field(description="is_bot from telegram")
    telegram_is_premium:     Optional[bool]       = Field(description="is_premium from telegram")
    telegram_language_code:  Optional[str]        = Field(description="language_code from telegram")

    @classmethod
    def from_telegram_user(cls, user: telegram.User) -> "Message":
        return User(username              = user.username,
                    first_name            = user.first_name,
                    last_name             = user.last_name,
                    telegram_id           = user.id,
                    telegram_is_bot       = user.is_bot,
                    telegram_is_premium   = user.is_premium,
                    telegram_lanuage_code = user.language_code)
    

class Message(SQLModel, table=True):
    id:               int           = Field(primary_key=True, description='Our unique message id')
    text:             str           = Field(max_length=4096, description='The message text')
    telegram_chat_id: Optional[int] = Field(description='The chat if from a telegram chat')
    user_id:          int           = Field(index=True, foreign_key='user.id', description='The user who sent the Message')
    created_at:       timestamp     = Field(index=True, default_factory=time, description='The epoch timestamp when the Message was created.')
    
    @classmethod
    def from_telegram_update(cls, update: telegram.Update, user : User) -> "Message":
        logger.info(f"Message from {user.id} {user.first_name} {user.last_name}: {update.message.text}")
        return Message(text=update.message.text, user_id=user.id, telegram_chat_id=update.message.chat.id)



    
class URL(SQLModel, table=True):
    id:             int       = Field(primary_key=True, description='Unique ID')
    url:            str       = Field(max_length=2048, description='The actual supplied URL')
    user_id:        int       = Field(index=True, foreign_key='user.id', description='The user who sent the URL')
    created_at:     timestamp = Field(default_factory=time, description='The epoch timestamp when this was created.')
                                      
                            
class UrlText(SQLModel, table=True):
    id:           int           = Field(primary_key=True, description="The text unique id.")
    url_id:       int           = Field(index=True, foreign_key="url.id", description="The usr this text was extracted from.")
    mechanism:    str           = Field(description="identifies which software mechanism exracted the text from the url")
    created_at:   timestamp     = Field(default_factory=time, description='The epoch timestamp when the url was crawled.')
    text:         str           = Field(max_length=65535, description="The readable text we managed to extract from the Url.")
    content:      Optional[str] = Field(max_length=65535, description="original html content")
    content_type: Optional[str] = Field(description="content type from http")

            
class UrlSummary(SQLModel, table=True):
    id:         int       = Field(primary_key=True, description="The summary unique id.")
    text_id:    int       = Field(index=True, foreign_key="urltext.id", description="The UrlText used to create the summary.")
    model:      str       = Field(description="The model used to produce this summary.")
    prefix:     str       = Field(max_length=8192, description="The prompt prefix used to create the summary.")
    summary:    str       = Field(max_length=8192, description="The summary we got back from the model.")
    created_at: timestamp = Field(default_factory=time, description='The epoch timestamp when this was created.')

    

class Completion(SQLModel, table=True):
    id:          int       = Field(primary_key=True, description="The completion unique id.")
    model:       str       = Field(description="model engine")
    temperature: int       = Field(description="requested temperature")
    prompt:      str       = Field(max_length=65535, description="The prompt used to generate the completion.")
    completion:  str       = Field(max_length=65535, description="The completion received from the model.")
    created_at:  timestamp = Field(default_factory=time, description='The epoch timestamp when this was created.')


    
class EmbeddingSource(str, enum.Enum):
    """
    The source of the text for embedding
    """
    message      = "message"
    url_summary  = "url_summary"
    

class Embedding(SQLModel, table=True):
    id:         int             = Field(default=None,
                                        primary_key=True,
                                        description='Unique database identifier for a given embedding vector.')
    collection: str             = Field(index=True, description='The name of the collection that holds this vector.')
    source:     EmbeddingSource = Field(sa_column=Column(Enum(EmbeddingSource)),
                                        description='The source of this embedding')
    source_id:  int             = Field(description='The memo that produced this embedding.')    
    model:      str             = Field(description="The model used to produce this embedding.")    
    vector:     List[float]     = Field(sa_column=Column(ARRAY(Float(24))),
                                        description='The embedding vector.')

    
