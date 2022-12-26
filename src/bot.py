"""
MassGPT 

A destructive distillation of mass communication.

Condition an LLM completion on a dynamically assembled subprompt context.

Telegram: @MassGPTbot
https://t.me/MassGPTbot

Copyright (C) 2022 William S. Kish

"""

import os
from sqlmodel import Session, select
from loguru import logger
from pydantic import BaseModel, Field
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters
import re

import openai
from db import engine
from models import *
from exceptions import *

import massgpt


# the bot app
bot = ApplicationBuilder().token(os.environ['MASSGPT_TELEGRAM_API_TOKEN']).build()


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
            tuser = update.message.from_user
            user =  User(username              = tuser.username,
                         first_name            = tuser.first_name,
                         last_name             = tuser.last_name,
                         telegram_id           = tuser.id,
                         telegram_is_bot       = tuser.is_bot,
                         telegram_is_premium   = tuser.is_premium,
                         telegram_lanuage_code = tuser.language_code)
            session.add(user)
            session.commit()
            session.refresh(user)
    return user



async def message(update: Update, tgram_context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle message received from user.
    Send back to the user the response text from the model.
    Handle exceptions by sending an error message to the user.    
    """
    user = get_telegram_user(update)
    text = update.message.text    
    logger.info(f'{user.id} {user.first_name} {user.last_name} {user.username} {user.telegram_id}: "{text}"')
    try:
        url = extract_url(text)
        if url:
            response = massgpt.summarize_url(user, url)
        else:
            response = massgpt.receive_message(user, text)
        await update.message.reply_text(response)
    except openai.error.ServiceUnavailableError:
        await update.message.reply_text("The OpenAI server is overloaded.")
    except ExtractException:
        await update.message.reply_text("Unable to extract text from url.")
    except MinimumTokenLimit:
        await update.message.reply_text("Message too short; please send a longer message.")
    except MaximumTokenLimit:
        await update.message.reply_text("Message too large; please send a shorter message.")
    except Exception as e:
        logger.exception("error processing message")
        await update.message.reply_text("An exceptional condition occured.")
        

bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message))







    
    
    
async def command(update: Update, tgram_context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle command from user
    context - Respond with the current chat context
    url - Summarize a url and add the summary to the chat context
    """
    user = get_telegram_user(update)
    text = update.message.text
    
    logger.info(f'{user.id} {user.first_name} {user.last_name} {user.username} {user.telegram_id}: "{text}"')
    

    if text == '/context':
        await update.message.reply_text("The current context:")            
        for msg in massgpt.current_context():
            await update.message.reply_text(msg)
        return
    #elif text == '/prompts':
    #    await update.message.reply_text("Current Prompt Stack:")
    #    await update.message.reply_text("(PREPROMPT) " + PREPROMPT )
    #    await update.message.reply_text("  [Context as shown by /context]")
    #    await update.message.reply_text("(PENULTIMATE_PROMPT) " + PENULTIMATE_PROMPT)
    #    await update.message.reply_text("  [Most recent message from user]")
    #    return
    elif text[:5] == '/url ':
        try:
            url = extract_url(update)            
            response = massgpt.summarize_url(user, url)
            await update.message.reply_text(response)
        except openai.error.ServiceUnavailableError:
            await update.message.reply_text("The OpenAI server is overloaded.")
        except ExtractException:
            await update.message.reply_text("Unable to extract text from url.")
        except Exception as e:
            logger.exception("error processing message")
            await update.message.reply_text("An exceptional condition occured.")
        return    
    await update.message.reply_text("Send me a message and I will assemble a collection of recent or related messages into a GPT prompt context and prompt your message against that dynamic context, sending you the GPT response.  Send /context to see the current prompt context. Send '/url <url>' to add a summary of the url to the context. Consider this to be a public chat and please maintain a kind and curious standard.")

    
bot.add_handler(MessageHandler(filters.COMMAND, command))


logger.info("run_polling")
bot.run_polling()
