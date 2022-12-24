"""
MassGPT 

A destructive distillation of mass communication.

Condition an LLM completion on a dynamically assembled subprompt context.

Telegram: @MassGPTbot
https://t.me/MassGPTbot

Copyright (C) 2022 William S. Kish

"""

import os
from sqlmodel import Field, SQLModel, Column, ARRAY, Float, Session, select
from loguru import logger
from pydantic import BaseModel, Field
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
import openai
from sentence_transformers import SentenceTransformer
import re
import urllib.parse
from extract import url_to_text, ExtractException
from db import engine
from models import *

from tokenizer import token_len

# General Prompt Strategy:
#  Upon reception of message from a user 999, compose the following prompt
#  based on recent messages received from other users:

from subprompt import SubPrompt


PREPROMPT = SubPrompt.from_str( \
"""You are MassGPT, and this is a fun experiment. \
Instruction: Different users are sending you messages. \
They can not communicate with each other directly. \
Any user to user message must be relayed through you. \
Pass along any interesting message. \
If a user expresses interest in a topic discussed here, \
respond to them based on what you read here. \
Users have recently said the following to you:""")


#  "User 123 wrote: ABC is the greatest thing ever"
#  "User 234 wrote: ABC is cool but i like GGG more"
#  "User 345 wrote: DDD is the best ever!"
#  etc
PENULTIMATE_PROMPT = SubPrompt.from_str( \
"""Instruction: Respond to the following user message considering the above context and Instruction:""")

#  "User 999 wrote:  What do folks think about ABC?"   # End of Prompt

# Then send resulting llm completion back to user 999 in response to his message



###
### Prompt Prefixen for URL summarization
###

# the PROMPT_PREFIX is prepended to the url content before sending to the language model
SUMMARIZE_PROMPT_PREFIX = SubPrompt.from_str("Provide a detailed summary of the following web content, including what type of content it is (e.g. news article, essay, technical report, blog post, product documentation, content marketing, etc). If the content looks like an error message, respond 'content unavailable'. If there is anything controversial please highlight the controversy. If there is something surprising, unique, or clever, please highlight that as well:")

# prompt prefix for Github Readme files
GITHUB_PROMPT_PREFIX = SubPrompt.from_str("Provide a summary of the following github project readme file, including the purpose of the project, what problems it may be used to solve, and anything the author mentions that differentiates this project from others:")

SUMMARIZE_MODEL_TEMPERATURE = 0.2    # temperature to use for URL summarization tasks


openai.api_key       = os.environ["OPENAI_API_KEY"]

OPENAI_ENGINE        = os.environ.get("OPENAI_ENGINE", "text-davinci-003")
MAX_CONTEXT_WINDOW   = 4097
MODEL_TEMPERATURE    = 0.54


MIN_RESPONSE_TOKENS = 256

MAX_CONTEXT_TOKENS = MAX_CONTEXT_WINDOW - MIN_RESPONSE_TOKENS - len(tokenizer(PREPROMPT + PENULTIMATE_PROMPT)['input_ids'])
MAX_SUBPROMPT_TOKENS  = 256   # limit of tokens from a single user message sub-prompt



## Embedding Config
ST_MODEL_NAME   =  'multi-qa-mpnet-base-dot-v1'
st_model        =  SentenceTransformer(ST_MODEL_NAME)


# the bot app
bot = ApplicationBuilder().token(os.environ['MASSGPT_TELEGRAM_API_TOKEN']).build()


class MessagePrompt(SubPrompt):
    @classmethod
    def from_msg(cls, msg: Message) -> "SubPrompt":
        # create actual subprompt for this users message
        text = f"User {msg.user_id} wrote to MassGPT: {msg.text}\n"
        # verify message is of acceptable length
        token_count = token_len(text)
        return MessagePrompt.from_text(text=text, MAX_SUBPROMPT_TOKENS)

class UrlSummaryPrompt(SubPrompt):    
    @classmethod
    def from_summary(cls, user: User, text : str) -> "SubPrompt":
        text = f"User {user.id} posted a link with the following summary:\n{text}\n"
        token_count = token_len(text)
        if token_count > MAX_SUBPROMPT_TOKENS + 20:  # allow slightly extra for subprompt overhead
            raise MessageTooLargeException
        return UrlSummaryPrompt.from_text(text=text)




class MessageTooLargeException(Exception):
    """
    Raise this exception if the user message is too large
    """

    

    
class  Context():
    """
    A context for assembling a large prompt context from recent user message subprompts
    """
    def __init__(self) -> "Context":
        self.tokens  = 0
        self._sub_prompts = []

    def add(self, sub_prompt : SubPrompt) -> None:
        # add new prompt to end of sub_prompts
        self._sub_prompts.append(sub_prompt)
        self.tokens += sub_prompt.tokens
        # remove oldest subprompts if over SUBCONTEXT limit
        while self.tokens > MAX_CONTEXT_TOKENS:
            self.tokens -= self._sub_prompts.pop(0).tokens
        logger.warning(self.tokens)
            
    def sub_prompts(self) -> list[SubPrompt]:
        return self._sub_prompts

    
##    
### maintain a single global context (for now)
##
context = Context()


def build_prompt(msg_subprompt : SubPrompt) -> str:
    """
    return prompt_text
    """
    prompt = PREPROMPT
    # add previous message context
    for sub in context.sub_prompts():
        prompt += sub.text
    # add most recent user message after penultimate prompt
    prompt += PENULTIMATE_PROMPT + msg_subprompt.text + "\n"
    return prompt


def extract_url(update: Update):
    try:
        return re.search("(?P<url>https?://[^\s]+)", update.message.text).group("url")
    except AttributeError:
        return None



def get_telegram_user(update : Update) -> User:
    """
    return database User object of the sender of a telegram message
    Create the user if it didn't previously exist.
    """
    with Session(engine) as session:
        user = session.exec(select(User).where(User.telegram_id == update.message.from_user.id)).first()
        if not user:
            user = User.from_telegram_user(update.message.from_user)
            session.add(user)
            session.commit()
            session.refresh(user)        
    return user


def process_user_message(update: Update, tgram_context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    process the user's message returning the model response string or raising exceptions on error
    """
    # Get user
    user = get_telegram_user(update)
    
    if extract_url(update):
        return summarize_url(update)

    with Session(engine) as session:        
        # persist msg to database so we can regain recent msg context after pod restart
        msg = Message.from_telegram_update(update, user)
        msg_subprompt = SubPrompt.from_msg(msg)
        session.add(msg)
        session.commit()
        session.refresh(msg)
        # embedding should move to background work queue
        t0 = time()
        embedding = Embedding(source     = EmbeddingSource.message,
                              source_id  = msg.id,
                              collection = ST_MODEL_NAME,
                              model      = ST_MODEL_NAME,
                              vector     = st_model.encode(msg.text))
        logger.info(f"embedding dt: {time()-t0}")
        session.add(embedding)
        session.commit()
        session.refresh(msg)

    # build final aggregate prompt
    prompt = build_prompt(msg_subprompt)
    token_count = token_len(prompt)
    
    logger.info(f"final prompt token_count: {token_count}  chars: {len(prompt)}")
    
    resp = openai.Completion.create(engine      = OPENAI_ENGINE,
                                    prompt      = prompt,
                                    temperature = MODEL_TEMPERATURE,
                                    max_tokens  = MAX_CONTEXT_WINDOW-token_count)
    
    response = resp.choices[0].text
    logger.info(response)

    # log completion to db
    with Session(engine) as session:        
        session.add(Completion(model       = OPENAI_ENGINE,
                               temperature = MODEL_TEMPERATURE,
                               prompt      = prompt,
                               completion  = response))
        session.commit()

    # add the new user message to the list of recent subprompts
    context.add(msg_subprompt)
    
    return response

    

async def message(update: Update, tgram_context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle message received from user.
    Send back to the user the response text from the model.
    Handle exceptions by sending an error message to the user.    
    """
    user = update.message.from_user
    logger.info(f'"{update.message.text}" {user.id} {user.first_name} {user.last_name} {user.username}')
    try:
        response = process_user_message(update, context)
        await update.message.reply_text(response)
    except ExtractException:
        await update.message.reply_text("Unable to extract text from url.")
    except MessageTooLargeException:
        await update.message.reply_text("Message too large; please send a shorter message.")
    except Exception as e:
        logger.exception("error processing message")
        await update.message.reply_text("An exceptional condition occured.")
        

bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message))




async def send_context(update: Update) -> None:
    """
    Send the current context to user who requested it via /context command
    """
    await update.message.reply_text("The current context:")            
    for sub in context.sub_prompts():
        await update.message.reply_text(sub.text)
    await update.message.reply_text(f"{context.tokens} tokens")


def summarize_url(update: Update) -> None:
    """
    summarize a user-supplied URL
    """
    # Get user
    user = get_telegram_user(update)

    # check if message contains a URL
    # if so extract and summarize the contents
    url = extract_url(update)
    if not url:
        return "Unable to parse URL"

    text = url_to_text(url)
    with Session(engine) as session:
        db_url = URL(url=url, user_id = user.id)
        session.add(db_url)
        session.commit()
        session.refresh(db_url)
        urltext = UrlText(url_id    = db_url.id,
                          mechanism = "url_to_text",
                          text      = text)
        session.add(urltext)
        session.commit()
        session.refresh(urltext)
    # use different prompt prefix for github versus normal web site
    if urllib.parse.urlparse(url).netloc == 'github.com':
        prefix = GITHUB_PROMPT_PREFIX
    else:
        prefix = SUMMARIZE_PROMPT_PREFIX

    # compose final prompt and truncate
    prompt = prefix + text
    prompt.truncate(MAX_CONTEXT_WINDOW - MAX_SUBPROMPT_TOKENS)

    resp = openai.Completion.create(engine      = OPENAI_ENGINE,
                                    prompt      = prompt,
                                    temperature = SUMMARIZE_MODEL_TEMPERATURE,
                                    max_tokens  = MAX_SUBPROMPT_TOKENS)
    response = resp.choices[0].text

    # log UrlSummary to db
    with Session(engine) as session:
        url_summary = UrlSummary(text_id = urltext.id,
                                 user_id = user.id,
                                 model   = OPENAI_ENGINE,
                                 prefix  = prefix,
                                 summary = response)
        session.add(url_summary)
        session.commit()
        session.refresh(url_summary)
        
        # embedding should move to background work queue
        t0 = time()
        embedding = Embedding(source     = EmbeddingSource.url_summary,
                              source_id  = url_summary.id,
                              collection = ST_MODEL_NAME,
                              model      = ST_MODEL_NAME,
                              vector     = st_model.encode(response))
        logger.info(f"embedding dt: {time()-t0}")
        session.add(embedding)
        
        session.commit()

    # add the summary to recent context    
    context.add(SubPrompt.from_summary(user=user, text=response))

    logger.info(response)
    # send the text summary to the user as FYI
    return response


    
    
async def command(update: Update, tgram_context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle command from user
    context - Respond with the current chat context
    url - Summarize a url and add the summary to the chat context
    config - Show the user's current preprompt, penultimate prompt, and temperature config
    temperature - Set a custom temperature for model completion
    preprompt - Set a custom preprompt
    penprompt - Set a custom penultimate prompt
    reset - Reset 
    """
    user = update.message.from_user
    logger.info(f'"{update.message.text}" {user.id} {user.first_name} {user.last_name} {user.username}')
    text = update.message.text
    if text == '/context':
        await send_context(update)
        return
    elif text == '/prompts':
        await update.message.reply_text("Current Prompt Stack:")
        await update.message.reply_text("(PREPROMPT) " + PREPROMPT )
        await update.message.reply_text("  [Context as shown by /context]")
        await update.message.reply_text("(PENULTIMATE_PROMPT) " + PENULTIMATE_PROMPT)
        await update.message.reply_text("  [Most recent message from user]")
        return
    elif text[:5] == '/url ':
        try:
            response = summarize_url(update)
            await update.message.reply_text(response)
        except ExtractException:
            await update.message.reply_text("Unable to extract text from url.")
        except Exception as e:
            logger.exception("error processing message")
            await update.message.reply_text("An exceptional condition occured.")
        return    
    await update.message.reply_text("Send me a message and I will assemble a collection of recent or related messages into a GPT prompt context and prompt your message against that dynamic context, sending you the GPT response.  Send /context to see the current prompt context. Send '/url <url>' to add a summary of the url to the context. Consider this to be a public chat and please maintain a kind and curious standard.")


    
bot.add_handler(MessageHandler(filters.COMMAND, command))


def load_context_from_db():
    logger.info('load_context_from_db')    
    with Session(engine) as session:
        for msg in session.exec(select(Message).order_by(Message.id.desc())):
            context.add(SubPrompt.from_msg(msg))
            if context.tokens > MAX_CONTEXT_TOKENS - MAX_SUBPROMPT_TOKENS:
                break
    context._sub_prompts.reverse()

load_context_from_db()

logger.info("run_polling")
bot.run_polling()
