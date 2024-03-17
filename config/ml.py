import transformers
from loguru import logger

from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
import torch

transformers.logging.set_verbosity_info()

logger.info("downloading sentence transformer")
sentence_model = SentenceTransformer('intfloat/multilingual-e5-base')

logger.info("downloading LLM")
tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B', eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)