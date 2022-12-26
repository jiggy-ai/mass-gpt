#  Completion Abstraction
#  Copyright(C) 2022 William S. Kish


import os
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from models import Completion
from subprompt import SubPrompt

from db import engine
from exceptions import *


class CompletionLimits(BaseModel):
    """
    specification for limits for:
    min prompt tokens
    min completion tokens
    max completion tokens
    given a max_content tokens
    """
    max_context      : int
    min_prompt       : int
    min_completion   : int
    max_completion   : int    

    def max_completion_tokens(self, prompt : SubPrompt) -> int:
        """
        returns the maximum completion tokens available given the max_context limit
        and the actual number of tokens in the prompt
        raises MinimumTokenLimit or MaximumTokenLimit exceptions if the prompt
        is too small or too big.
        """
        if prompt.tokens < self.min_prompt:
            raise MinimumTokenLimit
        if prompt.tokens > self.max_prompt_tokens():
            raise MaximumTokenLimit
        max_available_tokens = self.max_context - prompt.tokens
        if max_available_tokens > self.max_completion:
            return self.max_completion
        return max_available_tokens

    def max_prompt_tokens(self) -> int:
        """
        return the maximum prompt size in tokens
        """
        return self.max_context - self.min_completion

    

class CompletionTask:
    """
    A LLM completion task that shares a particular llm configuration and prompt/completion limit structure. 
    """
    def __init__(self,
                 limits : CompletionLimits,                 
                 temperature : float,
                 model  : str,
                 inference_func) -> "CompletionTask" :
        self.limits = limits
        self.temperature = temperature
        self.model = model
        self.ifunc = inference_func

    def max_prompt_tokens(self):
        return self.limits.max_prompt_tokens()

    def completion(self, prompt : SubPrompt) -> Completion:
        """
        prompt the model with the specified prompt and return the resulting Completion
        """
        # check prompt limits and return max completion size to request
        # given the size of the prompt and the configured limits
        max_completion = self.limits.max_completion_tokens(prompt)

        # perform the completion inference
        response = self.ifunc(model                 = self.model,
                              prompt                = str(prompt),
                              temperature           = self.temperature,
                              max_completion_tokens = max_completion) 
        
        # save Completion object in db
        with Session(engine) as session:
            completion = Completion(model       = self.model,
                                    temperature = self.temperature,
                                    prompt      = str(prompt),
                                    completion  = response)
            session.add(completion)
            session.commit()
            session.refresh(completion)
        return completion
