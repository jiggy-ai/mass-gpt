# used at docker build time to pull model cache into container

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

from sentence_transformers import SentenceTransformer
st_model_name = 'multi-qa-mpnet-base-dot-v1'
st_model = SentenceTransformer(st_model_name)

#import whisper
#whisper_model = whisper.load_model("large")

print("HF done")
