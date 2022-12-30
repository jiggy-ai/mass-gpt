# GPT3 specific Completions
# Copyright (C) 2022 William S. Kish

import os
from loguru import logger
from time import sleep

import completion
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

OPENAI_COMPLETION_MODELS = ["text-davinci-003", "text-davinci-002", "text-davinci-001"]


def CompletionLimits(min_prompt:int,
                     min_completion:int,
                     max_completion:int,
                     max_context: int = 4097) -> completion.CompletionLimits:
    """
    CompletionLimits specific to GPT3 models with max_context of 4097 tokens
    """
    assert(max_context <= 4097)
    return completion.CompletionLimits(max_context    = max_context,   
                                       min_prompt     = min_prompt,
                                       min_completion = min_completion,
                                       max_completion = max_completion)




class GPT3CompletionTask(completion.CompletionTask):
    """
    An OpenAI GP3-class completion task implemented using OpenAI API
    """
    RETRY_COUNT = 5
    
    def __init__(self,
                 limits      : completion.CompletionLimits,
                 temperature : float = 1,
                 top_p       : float = 1,
                 stop        : list[str] = None,
                 model       : str = 'text-davinci-003') -> "GPT3CompletionTask":
        
        assert(model in OPENAI_COMPLETION_MODELS)

        if model in  ["text-davinci-003", "text-davinci-002"]:
            assert(limits.max_context <= 4097)
        else:
            assert(limits.max_context <= 2048)

        self.stop  = stop
        self.top_p = top_p
        self.temperature = temperature
        
        super().__init__(limits = limits,
                         model  = model)

        
    def _completion(self,
                    prompt                : str,                        
                    max_completion_tokens : int) -> str :
        """
        perform the actual completion via openai api
        returns the completion text string
        """
        def completion():
            resp = openai.Completion.create(engine      = self.model,
                                            prompt      = prompt,
                                            temperature = self.temperature,
                                            top_p       = self.top_p,
                                            stop        = self.stop,
                                            max_tokens  = max_completion_tokens)
            return resp.choices[0].text
            
        for i in range(GPT3CompletionTask.RETRY_COUNT):
            try:
                return completion()
            except openai.error.ServiceUnavailableError:
                logger.warning("openai ServiceUnavailableError")
                if i == RETRY_COUNT-1:
                    raise
                sleep((i+1)*.1)
            except Exception as e:
                logger.exception("_completion")
                raise
            

    
