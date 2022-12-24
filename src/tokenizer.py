

from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def token_len(text : str) -> int:
    """
    return number of tokens in text per gpt2 tokenizer
    """
    return len(tokenizer(text)['input_ids'])


