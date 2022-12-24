#  Begining of a framework for Jigits, which are widgets that allow LLMs to interact with code
#  Copyright (C) 2022 William S. Kish


from subprompt import SubPrompt


class Jigit(BaseModel):

    id:      str          # unique ID for a particular Jigit
    prompt:  SubPrompt    # The subprompt to include to activate this jidget


    def handler(completion: str):
        pass
    
    
    def process_completion(self, completion: str):
        if self.id in completion:
            self.handler(completion)
