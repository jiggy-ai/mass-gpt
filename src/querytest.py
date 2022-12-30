from loguru import logger
from gpt3 import GPT3CompletionTask, CompletionLimits
from exceptions import *
from models import Completion

from subprompt import SubPrompt

import wikipedia
    

class SubjectQueryTask(GPT3CompletionTask):
    def __init__(self,
                 min_completion=100,
                 max_completion=100) -> "SubjectQueryTask":
        
        limits = CompletionLimits(min_prompt     = 0,
                                  min_completion = min_completion,
                                  max_completion = max_completion)

        
        super().__init__(limits      = limits,
                         temperature = .1,
                         model      = 'text-davinci-003')

    def completion(self,
                   query : str) -> Completion:

        prompt = SubPrompt("What is the primary topic of this questions:")
        prompt += query
        print(prompt)
        resp = super().completion(prompt)
        print(str(resp))
        return resp


        
class WikipediaQueryTask(GPT3CompletionTask):
    """
    Generated message response completions based on dynamic history of recent messages and most used message
    """
    TEMPERATURE = 0.0
    
    def __init__(self,
                 wikipedia_page:str, 
                 min_completion=100,
                 max_completion=100) -> "WikipediaQueryTask":
        
        limits = CompletionLimits(min_prompt     = 0,
                                  min_completion = min_completion,
                                  max_completion = max_completion)

        self.page = wikipedia_page
        page = wikipedia.page(self.page)
        self.content = page.content
        
        super().__init__(limits      = limits,
                         temperature = 0,
                         model      = 'text-davinci-003')

                 
    def completion(self,
                   query : str) -> Completion:
        """
        return completion for the provided subprompts
        """

        prompt = SubPrompt(f"Respond to the final question regarding this info on '{self.page}' using only the information provided here in the following text:")
        final = SubPrompt(query)
        available_tokens = self.max_prompt_tokens() - (prompt + final).tokens - 1
        article = SubPrompt(self.content, max_tokens=available_tokens, truncate=True)
        prompt += article
        prompt += final
        print(prompt)
        resp = super().completion(prompt)
        print(str(resp))
        return resp

query_task = WikipediaQueryTask("rocket engine", 200, 400)



