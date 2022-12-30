# GPT3 specific Completions
# Copyright (C) 2022 William S. Kish

import os
from loguru import logger
from time import sleep

import completion
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

OPENAI_COMPLETION_MODELS = ["text-davinci-003", "text-davinci-002", "text-davinci-001"]


def CompletionLimits(min_prompt:int, min_completion:int, max_completion:int) -> completion.CompletionLimits:
    """
    CompletionLimits specific to GPT3 models with max_context of 4097 tokens
    """
    return completion.CompletionLimits(max_context    = 4097,   ## XXX 001 has half this XXX
                                       min_prompt     = min_prompt,
                                       min_completion = min_completion,
                                       max_completion = max_completion)



class GPT3CompletionTask(completion.CompletionTask):

    RETRY_COUNT = 5
    
    def __init__(self,
                 limits      : completion.CompletionLimits,
                 temperature : float,
                 model       : str = 'text-davinci-003') -> "GPT3CompletionTask":
        
        assert(model in OPENAI_COMPLETION_MODELS)
        assert(limits.max_context <= 4097)

        super(CompletionTask, self).__init__(limits         = limits,
                                             temperature    = temperature,
                                             model          = model,
                                             inference_func = inference_func)

    def _completion(self,
                    prompt                : str,                        
                    max_completion_tokens : int) -> str :
        """
        perform the actual completion via openai api
        returns the completion text string
        """
        for i in range(GPT3CompletionTask.RETRY_COUNT):
            try:
                resp = openai.Completion.create(engine      = self.model,
                                                prompt      = prompt,
                                                temperature = self.temperature,
                                                max_tokens  = max_completion_tokens)
                break
            except openai.error.ServiceUnavailableError:
                if i == RETRY_COUNT-1:
                    raise
                sleep((i+1)*.1)
        #logger.info(resp)
        return resp.choices[0].text


    
