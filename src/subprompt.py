#
#  SubPrompt class that assists with keeping track of token counts and 
#  efficiently combining SubPrompts
#
#  Copyright (C) 2022 William S. Kish

import os
from pydantic import BaseModel, Field
from tokenizer import token_len
from typing import Optional
from exceptions import *


# have found that marking truncated text as truncated stops the model from trying
# to complete the missing text instead of summarizing it as requested
TRUNCATED = "<TRUNCATED>"
TRUNCATED_LEN = token_len(TRUNCATED)

class SubPrompt:
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

    def truncate(self, max_tokens, precise=False):
        if precise == True:
            raise Exception("precise truncation is not yet implemented")        
        # crudely truncate longer texts to get it back down to approximately the target max_tokens
        # TODO: find precise truncation point using multiple calls to token_len()
        # TODO: consider option to truncating at sentence boundaries.
        if self.tokens <= max_tokens:
            return
        split_point = int(len(self.text) * (max_tokens-TRUNCATED_LEN) / self.tokens)
        while not self.text[split_point].isspace():            
            split_point -= 1
        self.text = self.text[:split_point] + TRUNCATED
        self.tokens = token_len(self.text)
        if self.tokens > max_tokens:
            self.truncate(max_tokens*.95)
        
        
    def __init__(self, text: str, max_tokens=None, truncate=False, precise=False, tokens=None) -> "SubPrompt":
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
        if tokens is None:
            tokens = token_len(text)
        self.text = text
        self.tokens = tokens
        if max_tokens is not None and tokens > max_tokens:
            if not truncate: 
                raise MaximumTokenLimit
            self.truncate(max_tokens, precise=precise)

    

    def __len__(self) -> int:        
        return self.tokens
    

    def __add__(self, o) -> "SubPrompt":
        """
        Combine the token strings and token counts with a newline character in between them.
        This will occasionally overestimate the combined token count by 1 token,
        which is acceptable for our intended use.
        """
        if isinstance(o, str):
            o = SubPrompt(o)
        
        return SubPrompt(text   = self.text + "\n" + o.text,
                       tokens = self.tokens  + 1 + o.tokens)
    
    def __str__(self):
        return self.text


    
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
        sp1 = SubPrompt(c1)
        sp2 = SubPrompt(c2)
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





