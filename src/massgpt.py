#  MassGPT app
#
#  Copyright (C) 2022 William S. Kish

from sqlmodel import Session, select, delete
from sentence_transformers import SentenceTransformer
import urllib.parse

from db import engine

import gpt3

from exceptions import *
from models import *

from extract import url_to_text
from subprompt import SubPrompt

from hashlib import md5

###
###  Various Specialized SubPrompts
###

class MessageSubPrompt(SubPrompt):
    """
    SubPrompt Context for a user-generated message
    """
    MAX_TOKENS = 300
    
    @classmethod
    def from_msg(cls, msg: Message) -> "SubPrompt":
        # create user message specific subprompt
        username = md5(str(msg.user_id).encode("utf-8")).hexdigest()[:5]
        text = f"user-{username} wrote to MassGPT: {msg.text}"
        return MessageSubPrompt(text=text, max_tokens=MessageSubPrompt.MAX_TOKENS)
            

class MessageResponseSubPrompt(SubPrompt):
    """
    SubPrompt Context for a user-generated message
    """
    @classmethod
    def from_msg_completion(cls, msp: MessageSubPrompt, comp: Completion) -> "SubPrompt":
        return msp # + f"MassGPT responded: {comp.completion}"
            
    
    
class UrlSummarySubPrompt(SubPrompt):
    """
    SubPrompt Context for a user-requested URL Summary
    """
    @classmethod
    def from_summary(cls, user: User, text : str) -> "SubPrompt":
        text = f"User {user.id} posted a link with the following summary:\n{text}\n"
        # don't need to specify max _tokens here since the summary is a model output
        # that is regulated through the user_summary_limits
        return UrlSummarySubPrompt(text=text)


class SummarySubPrompt(SubPrompt):
    """
    SubPrompt Context for a system-generated summary
    """
    @classmethod
    def from_msg(cls, text : str) -> "SubPrompt":
        text = f"Here is a summary of previous discussions for reference: {text}"
        # don't need to specify max _tokens here since the summary is a model output
        # that is regulated through the msg_summary_limits        
        return SummarySubPrompt(text=text)





class MassGPTMessageTask(gpt3.GPT3CompletionTask):
    """
    Generated message response completions based on dynamic history of recent messages and most used message
    """
    TEMPERATURE = 0.4
    
    # General Prompt Strategy:
    #  Upon reception of message from a user 999, compose the following prompt
    #  based on recent messages received from other users:

    PREPROMPT = SubPrompt( \
"""You are MassGPT, and this is a fun experiment. \
You were built by Jiggy AI using OpenAI text-davinci-003. \
Instruction: Different users are sending you messages. \
They can not communicate with each other directly. \
Any user to user message must be relayed through you. \
Pass along any interesting message. \
Try not to repeat yourself. \
Ask users questions if you are not sure what to say. \
If a user expresses interest in a topic discussed here, \
respond to them based on what you read here. \
Users have recently said the following to you:""")


    #  "User 123 wrote: ABC is the greatest thing ever"
    #  "User 234 wrote: ABC is cool but i like GGG more"
    #  "User 345 wrote: DDD is the best ever!"
    #  etc
    PENULTIMATE_PROMPT = SubPrompt("Instruction: Respond to the following user message considering the above context and Instruction:")
    #  "User 999 wrote:  What do folks think about ABC?"   # End of Prompt
    FINAL_PROMPT = SubPrompt("MassGPT responded:")
    # Then send resulting llm completion back to user 999 in response to his message

    def __init__(self) -> "MassGPTMessageTask":
        limits = gpt3.CompletionLimits(min_prompt     = 0,
                                       min_completion = 300,
                                       max_completion = 400)
        
        super().__init__(limits      = limits,
                         temperature = MassGPTMessageTask.TEMPERATURE,
                         model      = 'text-davinci-003')

                 
    def completion(self,
                   recent_msgs : MessageResponseSubPrompt,
                   user_msg    : MessageSubPrompt) -> Completion:
        """
        return completion for the provided subprompts
        """
        prompt = MassGPTMessageTask.PREPROMPT
        final_prompt  = MassGPTMessageTask.PENULTIMATE_PROMPT
        final_prompt += user_msg
        final_prompt += MassGPTMessageTask.FINAL_PROMPT

        logger.info(f"overhead tokens: {(prompt + final_prompt).tokens}")
        
        available_tokens = self.max_prompt_tokens() - (prompt + final_prompt).tokens
        logger.info(f"available_tokens: {available_tokens}")
        # assemble list of most recent_messages up to available token limit
        reversed_subs = []
        # add previous message context
        for sub in reversed(recent_msgs):
            if available_tokens - sub.tokens -1 < 0:
                break
            reversed_subs.append(sub)
            available_tokens -= sub.tokens + 1

        for sub in reversed(reversed_subs):
            prompt += sub
            
        prompt += final_prompt
        # add most recent user message after penultimate prompt
        logger.info(f"final prompt tokens: {prompt.tokens}  max{self.max_prompt_tokens()}")

        logger.info(f"final prompt token_count: {prompt.tokens}  chars: {len(prompt.text)}")
        
        return super().completion(prompt)




msg_response_task   = MassGPTMessageTask()
                                          



class UrlSummaryTask(gpt3.GPT3CompletionTask):
    """
    A Factory class to dynamically compose a prompt context for URL Summary task
    """
    
    # the PROMPT_PREFIX is prepended to the url content before sending to the language model
    SUMMARIZE_PROMPT_PREFIX = SubPrompt("Provide a detailed summary of the following web content, including what type of content it is (e.g. news article, essay, technical report, blog post, product documentation, content marketing, etc). If the content looks like an error message, respond 'content unavailable'. If there is anything controversial please highlight the controversy. If there is something surprising, unique, or clever, please highlight that as well:")

    # prompt prefix for Github Readme files
    GITHUB_PROMPT_PREFIX = SubPrompt("Provide a summary of the following github project readme file, including the purpose of the project, what problems it may be used to solve, and anything the author mentions that differentiates this project from others:")

    TEMPERATURE = 0.2

    def __init__(self) -> "UrlSummaryTask":
        limits = gpt3.CompletionLimits(min_prompt     = 40,
                                       min_completion = 300,
                                       max_completion = 600)
        
        super().__init__(limits      = limits,
                         temperature = UrlSummaryTask.TEMPERATURE,
                         model      = 'text-davinci-003')

    def prefix(self, url):
        if urllib.parse.urlparse(url).netloc == 'github.com':
            return UrlSummaryTask.GITHUB_PROMPT_PREFIX
        else:
            return UrlSummaryTask.SUMMARIZE_PROMPT_PREFIX


    def completion(self, url: str, url_text : str) -> Completion:
        """
        return prompt text to summary the following url and and url_text.
        The url is required in able to enable host-specific prompt strategy.
        For example a different prompt is used to summarize github repo's versus other web sites.
        """
        prompt = self.prefix(url) + url_text
        prompt.truncate(self.max_prompt_tokens())
        return super().completion(prompt)




url_summary_task   =  UrlSummaryTask()



## Embedding Config
ST_MODEL_NAME   =  'multi-qa-mpnet-base-dot-v1'
st_model        =  SentenceTransformer(ST_MODEL_NAME)



    
class  Context():
    """
    A context for assembling a large prompt context from recent user message subprompts
    """
    def __init__(self) -> "Context":
        self.tokens  = 0
        self._sub_prompts = []

    def push(self, sub_prompt : SubPrompt) -> bool:
        """
        Push sub_prompt onto begining of context.
        Used to recreate context in reverse order from database select
        raises MaximumTokenLimit when prompt context limit is exceeded
        """
        if self.tokens > msg_response_task.max_prompt_tokens():
            raise MaximumTokenLimit
        self._sub_prompts.insert(0, sub_prompt)
        self.tokens += sub_prompt.tokens
        
    def add(self, sub_prompt : SubPrompt) -> None:
        # add new prompt to end of sub_prompts
        self._sub_prompts.append(sub_prompt)
        self.tokens += sub_prompt.tokens
        # remove oldest subprompts if over prompt context limit exceeded
        while self.tokens > msg_response_task.max_prompt_tokens():
            self.tokens -= self._sub_prompts.pop(0).tokens
            
    def sub_prompts(self) -> list[SubPrompt]:
        return self._sub_prompts

    
##    
### maintain a single global context (for now)
##
context = Context()


    
    
def receive_message(user : User, text : str) -> str:
    """
    receive a message from the specified user.
    Return the message response
    """    
    logger.info(f"message from {user.id} {user.first_name} {user.last_name}: {text}")    
    with Session(engine) as session:        
        # persist msg to database so we can regain recent msg context after pod restart
        msg = Message(text=text, user_id=user.id)
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
    msg_subprompt = MessageSubPrompt.from_msg(msg)
    
    completion = msg_response_task.completion(context.sub_prompts(), msg_subprompt)
    logger.info(str(completion))
    
    # add the new user message to the global shared context
    rsp_subprompt = MessageResponseSubPrompt.from_msg_completion(msg_subprompt, completion)
    context.add(rsp_subprompt)

    # save response to database
    with Session(engine) as session:
        session.add(completion)
        session.commit()
        session.refresh(completion)
        session.add(Response(message_id=msg.id,
                             completion_id=completion.id))
        session.commit()
        session.refresh(completion)
        
    return str(completion)

    


def summarize_url(user : User,  url : str) -> str:
    """
    Summarize a url for a user.
    Return the URL summary, adding the summary to the current context
    """
    # check if message contains a URL
    # if so extract and summarize the contents
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

    completion = url_summary_task.completion(url, text)

    summary_text = str(completion)
    
    with Session(engine) as session:
        session.add(completion)
        session.commit()
        session.refresh(completion)
        
        url_summary = UrlSummary(text_id = urltext.id,
                                 user_id = user.id,
                                 model   = url_summary_task.model,
                                 prefix  = url_summary_task.prefix(url).text,
                                 summary = summary_text)
        session.add(url_summary)
        session.commit()
        session.refresh(url_summary)
        
        # embedding should move to background work queue
        t0 = time()
        embedding = Embedding(source     = EmbeddingSource.url_summary,
                              source_id  = url_summary.id,
                              collection = ST_MODEL_NAME,
                              model      = ST_MODEL_NAME,
                              vector     = st_model.encode(summary_text))
        logger.info(f"embedding dt: {time()-t0}")
        session.add(embedding)
        
        session.commit()

    # add the summary to recent context    
    context.add(UrlSummarySubPrompt.from_summary(user=user, text=summary_text))

    logger.info(summary_text)
    # send the text summary to the user as FYI
    return summary_text



def current_prompts() -> str:

    prompt  = "Current prompt stack:\n\n"
    prompt += MassGPTMessageTask.PREPROMPT.text +"\n\n"
    prompt +=  "[Context as shown by /context]\n\n"
    prompt +=  MassGPTMessageTask.PENULTIMATE_PROMPT.text + "\n\n"
    prompt +=  "[Most recent message from user]"

    return prompt
    


def current_context(max_len=4096) -> str:
    """
    iterator returning max_len length strings of current context
    """
    size = 0
    text = ""
    for sub in context.sub_prompts():
        if len(sub.text) + size > max_len:
            yield text
            text = ""
            size = 0
        text += sub.text + "\n"
        size += len(sub.text) + 1
    if text:
        yield text
    yield f"{context.tokens} tokens"

    
    
def load_context_from_db():
    last = ""
    logger.info('load_context_from_db')    
    with Session(engine) as session:
        for msg in session.exec(select(Message).order_by(Message.id.desc())):
            msg_subprompt = MessageSubPrompt.from_msg(msg)
            if msg_subprompt.text == last: continue  # basic dedup
            last = msg_subprompt.text
            resp = session.exec(select(Response).where(Response.message_id == msg.id)).first()
            if not resp:
                try:
                    context.push(msg_subprompt)
                except MaximumTokenLimit:
                    break
                continue
            comp = session.exec(select(Completion).where(Completion.id == resp.completion_id)).first()
            if not comp: continue
            rsp_subprompt = MessageResponseSubPrompt.from_msg_completion(msg_subprompt, comp)
            try:
                context.push(rsp_subprompt)
            except MaximumTokenLimit:
                break

load_context_from_db()
