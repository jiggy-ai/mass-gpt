from loguru import logger
from gpt3 import GPT3CompletionTask, CompletionLimits
from exceptions import *
from models import Completion

from subprompt import SubPrompt

class MessageSubPrompt(SubPrompt):
    """
    SubPrompt Context for a user-generated message
    """
    MAX_TOKENS = 300    
    @classmethod
    def from_user_str(cls, username : str, msg: str) -> "SubPrompt":
        # create user message specific subprompt
        text = f"'user-{username}' wrote to MassGPT: {msg}"
        return MessageSubPrompt(text=text, max_tokens=MessageSubPrompt.MAX_TOKENS)
    

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
    

class GPTCopyTestTask(GPT3CompletionTask):
    """
    Generated message response completions based on dynamic history of recent messages and most used message
    """
    TEMPERATURE = 0.0
    
    # General Prompt Strategy:
    #  Upon reception of message from a user 999, compose the following prompt
    #  based on recent messages received from other users:

    #PREPROMPT = SubPrompt("Prepare to copy some of the following messages that users sent to MassGPT:")


    #  "User 123 wrote: ABC is the greatest thing ever"
    #  "User 234 wrote: ABC is cool but i like GGG more"
    #  "User 345 wrote: DDD is the best ever!"
    #  etc
    #PENULTIMATE_PROMPT = SubPrompt("Instruction: Respond to the following user message considering the above context and Instruction:")
    #  "User 999 wrote:  What do folks think about ABC?"   # End of Prompt
    #FINAL_PROMPT = SubPrompt("MassGPT responded:")
    # Then send resulting llm completion back to user 999 in response to his message

    def __init__(self) -> "GPTCopyTestTask":
        limits = CompletionLimits(min_prompt     = 0,
                                  min_completion = 300,
                                  max_completion = 400)
        
        super().__init__(limits      = limits,
                         temperature = GPTCopyTestTask.TEMPERATURE,
                         stop = ['###'],                         
                         model      = 'text-davinci-003')

                 
    def completion(self,
                   recent_msgs : MessageSubPrompt,
                   user_msg    : MessageSubPrompt) -> Completion:
        """
        return completion for the provided subprompts
        """
        #prompt = GPTCopyTestTask.PREPROMPT
        #final_prompt  = GPTCopyTestTask.PENULTIMATE_PROMPT
        #final_prompt += user_msg
        #final_prompt += GPTCopyTestTask.FINAL_PROMPT

        #logger.info(f"overhead tokens: {(prompt + final_prompt).tokens}")
        
        #available_tokens = self.max_prompt_tokens() - (prompt + final_prompt).tokens

        #logger.info(f"available_tokens: {available_tokens}")
        # assemble list of most recent_messages up to available token limit
        prompt = SubPrompt("")
        for sub in recent_msgs:
            prompt += sub
        prompt += user_msg
        #prompt += final_prompt
        # add most recent user message after penultimate prompt
        logger.info(f"final prompt tokens: {prompt.tokens}  max{self.max_prompt_tokens()}")

        logger.info(f"final prompt token_count: {prompt.tokens}  chars: {len(prompt.text)}")
        
        return super().completion(prompt)


msg_response_task   = GPTCopyTestTask()


#users = ['cat', 'dog', 'pig', 'cow', 'rocket', 'ocean', 'mountain', 'tree']
#users = ['cat', 'rocket']
users = list(range(10))

from random import choice


NUM_MESSAGES = 220
#NUM_MESSAGES = 150

data = []
messages = []
for i in range(NUM_MESSAGES):
    user = choice(users)
    msg  = f"This is message {i}"
    data.append((user, msg))
    msp = MessageSubPrompt.from_user_str(user, msg)
    messages.append(msp)
    print(msp)

print("=======================")
target_u = choice(users)


final_p = MessageSubPrompt.from_user_str(choice(users),
                                         f"Instruction: Copy all of the messages that 'user-{target_u}' wrote to MassGPT, only one message per line:")

final_p = SubPrompt(f"Instruction: Copy all of the messages that 'user-{target_u}' wrote to MassGPT, only one message per line:")
                                         

comp = msg_response_task.completion(messages, final_p)

print(comp.prompt)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(str(comp))
    

answers = [msg for user,msg in data if user==target_u]

correct = 0
results = comp.completion.rstrip().lstrip().split('\n')

answers = set([a.rstrip().lstrip().lower() for a in answers])
comps = set([c.rstrip().lstrip().lower() for c in results])

#print(answers)
#print(comps)
correct = answers.intersection(comps)


    
"""
for c , a in zip(results.split("\n"), answer):
    a = a.rstrip().lower()
    c = c.rstrip().lower()
    print(f"{a} | {c}")    
    if a in c:
        correct += 1
"""

precision = len(correct)/len(comps)
recall = len(correct)/len(answers)
print(f"precision {precision} \t recall {recall}")
