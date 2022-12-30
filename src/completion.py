#  Completion Abstraction
#  Copyright(C) 2022 William S. Kish


import os
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from models import Completion
from subprompt import SubPrompt

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
    This is a base class for a model-specific completion task.  A model-api-specific implemention must
    at a minimum implement the _completion() method.
    See gpt3.GPT3CompletionTask for an example implementation.
    """
    def __init__(self,
                 limits      : CompletionLimits,                 
                 model       : str) -> "CompletionTask" :
        
        self.limits = limits
        self.model = model

    def max_prompt_tokens(self) -> int:
        return self.limits.max_prompt_tokens()

    def limits(self) -> CompletionLimits:
        return self.limits

    def _completion(self,
                    prompt                : str,                        
                    max_completion_tokens : int) -> str :
        """
        perform the actual completion, returning the completion text string
        This should be implemented in a model-api-specific base class. 
        """
        pass

    def completion(self, prompt : SubPrompt) -> Completion:
        """
        prompt the model with the specified prompt and return the resulting Completion
        """        
        # check prompt limits and return max completion size to request
        # given the size of the prompt and the configured limits
        max_completion = self.limits.max_completion_tokens(prompt)
        
        # perform the completion inference
        response = self._completion(prompt                = str(prompt),
                                    max_completion_tokens = max_completion)

        completion = Completion(model       = self.model,
                                prompt      = str(prompt),
                                temperature = 0, # XXX  set this as model params?
                                completion  = response)
        return completion
