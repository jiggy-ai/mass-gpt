# GPT3 specific Completions
# Copyright (C) 2022 William S. Kish

import os
from loguru import logger
from time import sleep

import completion
import openai

openai.api_key       = os.environ["OPENAI_API_KEY"]

OPENAI_COMPLETION_MODELS = ["text-davinci-003", "text-davinci-002"]


def CompletionLimits(min_prompt:int, min_completion:int, max_completion:int):
    """
    CompletionLimits specific to GPT3 models with max_context of 4097 tokens
    """
    return completion.CompletionLimits(max_context    = 4097,
                                       min_prompt     = min_prompt,
                                       min_completion = min_completion,
                                       max_completion = max_completion)


RETRY_COUNT = 5

def inference_func(model                 : str,
                        prompt                : str,                        
                        temperature           : float,
                        max_completion_tokens : int) -> str :
    """
    openai inference func for use as CompletionTask inference_func
    """
    for i in range(RETRY_COUNT):
        try:
            resp = openai.Completion.create(engine      = model,
                                            prompt      = prompt,
                                            temperature = temperature,
                                            max_tokens  = max_completion_tokens)
            break
        except openai.error.ServiceUnavailableError:
            if i == RETRY_COUNT-1:
                raise
            sleep((i+1)*.1)
    return resp.choices[0].text



def CompletionTask(limits,
                   temperature,
                   model = 'text-davinci-003'):
    """
    return CompletionTask specific to GPT3 models
    """
    assert(model in OPENAI_COMPLETION_MODELS)
    assert(limits.max_context <= 4097)
    
    return completion.CompletionTask(limits         = limits,
                                     temperature    = temperature,
                                     model          = model,
                                     inference_func = inference_func)
                        


    
