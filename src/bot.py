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
from transformers import GPT2Tokenizer
import openai
from sentence_transformers import SentenceTransformer
import re
import urllib.parse
from extract import url_to_text, ExtractException
from db import engine
from models import *


# General Prompt Strategy:
#  Upon reception of message from a user 999, compose the following prompt
#  based on recent messages received from other users:
PREPROMPT   = "Different users are chatting but can't talk to each other directly. "
PREPROMPT  += "All messages are relayed through you. Please help the users "
PREPROMPT  += "communicate through you. Users have recently said the following:\n"
#  "User 123 wrote: ABC is the greatest thing ever"
#  "User 234 wrote: ABC is cool but i like GGG more"
#  "User 345 wrote: DDD is the best ever!"
#  etc
PENULTIMATE_PROMPT  = "\nRespond to the following user message considering the above context; "
PENULTIMATE_PROMPT += "also, forward any recent user message which is particularly interesting:\n"
#  "User 999 wrote:  What do folks think about ABC?"   # End of Prompt

# Then send resulting llm completion back to user 999 in response to his message



###
### Prompt Prefixen for URL summarization
###

# the PROMPT_PREFIX is prepended to the url content before sending to the language model
SUMMARIZE_PROMPT_PREFIX = "Provide a detailed summary of the following web content, including what type of content it is (e.g. news article, essay, technical report, blog post, product documentation, content marketing, etc). If the content looks like an error message, respond 'content unavailable'. If there is anything controversial please highlight the controversy. If there is something surprising, unique, or clever, please highlight that as well:\n"

# prompt prefix for Github Readme files
GITHUB_PROMPT_PREFIX = "Provide a summary of the following github project readme file, including the purpose of the project, what problems it may be used to solve, and anything the author mentions that differentiates this project from others:\n"

SUMMARIZE_MODEL_TEMPERATURE = 0.2    # temperature to use for URL summarization tasks


openai.api_key       = os.environ["OPENAI_API_KEY"]

OPENAI_ENGINE        = os.environ.get("OPENAI_ENGINE", "text-davinci-003")
MAX_CONTEXT_WINDOW   = 4097
MODEL_TEMPERATURE    = 0.5


tokenizer            = GPT2Tokenizer.from_pretrained("gpt2")

MAX_CONTEXT_TOKENS = 3200 - len(tokenizer(PREPROMPT + PENULTIMATE_PROMPT)['input_ids'])
MAX_SUBPROMPT_TOKENS  = 256   # limit of tokens from a single user message sub-prompt


## Embedding Config
ST_MODEL_NAME   =  'multi-qa-mpnet-base-dot-v1'
st_model        =  SentenceTransformer(ST_MODEL_NAME)


# the bot app
bot = ApplicationBuilder().token(os.environ['MASSGPT_TELEGRAM_API_TOKEN']).build()




class MessageTooLargeException(Exception):
    """
    Raise this exception if the user message is too large
    """

def num_tokens(text : str) -> int:
    """
    return number of tokens in text per gpt2 tokenizer
    """
    return len(tokenizer(text)['input_ids'])
    

class SubPrompt(BaseModel):
    """
    An subprompt crafted from a single user message or a summary of a user-supplied url.
    """
    text      : str     # the subprompt text
    tokens    : int     # the number of tokens in the subprompt text

    @classmethod
    def from_msg(cls, msg: Message) -> "SubPrompt":
        # create actual subprompt for this users message
        text = f"User {msg.user_id} wrote: {msg.text}\n"
        # verify message is of acceptable length
        token_count = num_tokens(msg.text)
        if token_count > MAX_SUBPROMPT_TOKENS:  
            raise MessageTooLargeException        
        return SubPrompt(text=text, tokens=token_count)

    @classmethod
    def from_summary(cls, user: User, text : str) -> "SubPrompt":
        text = f"User {user.id} posted a link with the following summary:\n{text}\n"
        token_count = num_tokens(text)
        if token_count > MAX_SUBPROMPT_TOKENS + 20:  # allow slightly extra for subprompt overhead
            raise MessageTooLargeException
        return SubPrompt(text=text, tokens=token_count)


    
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
    logger.info(prompt)
    return prompt


def extract_url(update: Update):
    try:
        return re.search("(?P<url>https?://[^\s]+)", update.message.text).group("url")
    except AttributeError:
        return None




def truncate_text(text: str, max_tokens : int = 3000) -> str:
    """
    truncate summmary text to ~max_tokens  tokens
    """
    # measure the text size in tokens
    token_count = num_tokens(text)
    if token_count > max_tokens:
        # crudely truncate longer texts to get it back down to approximately the target max_tokens
        # TODO: enhance to truncate at sentence boundaries using actual token counts
        split_point = int(len(text) * max_tokens / token_count)
        text = text[:split_point]        
    return text + "-"


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
    token_count = num_tokens(prompt)
    
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
    prompt = truncate_text(prefix + text)

    resp = openai.Completion.create(engine      = OPENAI_ENGINE,
                                    prompt      = prompt,
                                    temperature = SUMMARIZE_MODEL_TEMPERATURE,
                                    max_tokens  = MAX_SUBPROMPT_TOKENS)
    response = resp.choices[0].text

    # log UrlSummary to db
    with Session(engine) as session:
        session.add(UrlSummary(text_id = urltext.id,
                               user_id = user.id,
                               model   = OPENAI_ENGINE,
                               prefix  = prefix,
                               summary = response))
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
    """
    user = update.message.from_user
    logger.info(f'"{update.message.text}" {user.id} {user.first_name} {user.last_name} {user.username}')
    text = update.message.text
    if text == '/context':
        await send_context(update)
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
    await update.message.reply_text("Send the bot a message and it will assemble a collection of recent or related messages into a GPT prompt context and prompt your message against that dynamic context, sending you the GPT response.  Send /context to see the current prompt context. Send '/url <url>' to add a summary of the url to the context.")


    
bot.add_handler(MessageHandler(filters.COMMAND, command))


def load_context_from_db():
    logger.info('load_context_from_db')    
    with Session(engine) as session:
        for msg in session.exec(select(Message).order_by(Message.id.desc())):
            context.add(SubPrompt.from_msg(msg))
            if context.tokens > MAX_CONTEXT_TOKENS - MAX_SUBPROMPT_TOKENS:
                break
                    
    

load_context_from_db()

logger.info("run_polling")
bot.run_polling()
