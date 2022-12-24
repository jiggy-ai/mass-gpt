#
#  SubPrompt class that assists with keeping track of token counts and 
#  efficiently combining SubPrompts


import os
from pydantic import BaseModel, Field
from tokenizer import token_len
from typing import Optional


class MinimumTokenLimit(Exception):
    """
    The specified minimum token count has been exceeded
    """
    
class MaximumTokensExceeded(Exception):
    """
    The specified maximum token count has been exceeded
    """


def truncate_text(text: str, max_tokens: int) -> str:
    """
    truncate summmary text to ~max_tokens  tokens
    """
    # measure the text size in tokens
    

class SubPrompt(BaseModel):
    """
    A SubPrompt is a text string and associated token count for the string

    len(SubPrompt) returns the length of the SubPrompt in tokens

    SubPrompt1 + SubPrompt2 returns a new subprompt which contains the
    concatenated text of the 2 subprompts separated by "\n"
    and a mostly accurate token count.

    The combined token count is estimated (not computed) so can sometimes overestimate the
    actual token count by 1 token.  Tests on random strings show this occurs less
    than 1% of the time.
    """
    text   : str    # the subprompt text string
    tokens : int    # the number of tokens in the text string

    
    def truncate(self, max_tokens, precise=False):
        if precise == True:
            raise Exception("precise truncation is not yet implemented")        
        # crudely truncate longer texts to get it back down to approximately the target max_tokens
        # TODO: find precise truncation point using multiple calls to token_len()
        # TODO: consider option to truncating at sentence boundaries. 
        split_point = int(len(self.text) * max_tokens / self.tokens)
        while not self.text[split_point].isspace():
            split_point -= 1
        self.text = self.text[:split_point] 
        self.tokens = token_len(self.text)
        
        
    @classmethod
    def from_str(cls, text: str, max_tokens=None, truncate=False, precise=False) -> "SubPrompt":
        """
        Create a subprompt from the specified string.
        If max_tokens is specified, then the SubPrompt will be limited to max_tokens.
        The behavior when max_tokens is exceeded is controlled by truncate.
        MaximumTokenLimit exception raised if the text exceeds the specified max_tokens and truncate is False.
        If truncate is true then the text will be truncated to meet the limit.
        If precise is False then the truncation will be very quick but only approximate.
        If precise is True then the truncation will be slower but guaranteed to meet the max_tokens limit.
        """
        if precise == True:
            raise Exception("precise truncation is not yet implemented")
        tokens = token_len(text)
        sp = SubPrompt(text=text, tokens=tokens)
        if max_tokens is not None and tokens > max_tokens:
            if not truncate: 
                raiase Exception(MaximumTokenLimit)
            sp.truncate(max_tokens, precise=precise)
        return sp
    

    def __len__(self) -> int:
        return self.tokens

    def __add__(self, o) -> "SubPrompt":
        """
        Combine the token strings and token counts with a newline character in between them.
        This will occasionally overestimate the combined token count by 1 token,
        which is acceptable for our intended use.
        """
        if isinstance(o, str):
            o = SubPrompt.from_str(o)
        return SubPrompt(text   = self.text + "\n" + o.text,
                         tokens = self.tokens  + 1 + o.tokens)
    def __str__(self):
        return self.text



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
        returns the maximum completion tokens available given the model_max_context limit
        and the actual number of tokens in the prompt
        raises MinimumTokenLimit or MaximumTokenLimit exceptions if the prompt
        is too small or too big.
        """
        if prompt.tokens < self.min_prompt:
            raise MinimumTokenLimit
        if prompt.tokens > self.max_prompt_tokens():
            raise MaximumTokenLimit
        max_available_tokens = self.model_max_context - prompt.tokens
        if max_available_tokens > self.max_completion:
            return self.max_completion
        return max_available_tokens

    def max_prompt_tokens(self) -> int:
        """
        return the maximum prompt size in tokens
        """
        return self.model_max_context - self.min_completion

    
    
if __name__ == "__main__":
    """
    test with random strings
    """
    from string import ascii_lowercase, whitespace, digits
    from random import sample, randint

    chars = ascii_lowercase + whitespace + digits 

    def randstring(n):
        return "".join([sample(chars, 1)[0] for i in range(n)])

    count = 0
    for i in range(100000):
        c1 = randstring(randint(20, 100)) 
        c2 = randstring(randint(20, 100))
        print("c1", c1)
        print("c2", c2)
        sp1 = SubPrompt.from_str(c1)
        sp2 = SubPrompt.from_str(c2)
        print("sp1", sp1)
        print("sp2", sp2)

        print("len sp1:", len(sp1))
        print("len sp2:", len(sp2))        
        #assert(len(sp1) == token_len(sp1.text))
        #assert(len(sp2) == token_len(sp2.text))        

        sp3 = sp1 + sp2
        print("sp3", sp3)
        
        print("len sp3:", len(sp3))
        sp3len = token_len(sp3.text)
        print(sp3len)
        if len(sp3) != sp3len:
            count += 1
        assert(len(sp3) >= sp3len)
    print(count, "errors")
    # 651 errors out of 100000 on a typical run





