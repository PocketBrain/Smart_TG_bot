import transformers
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline

transformers.logging.set_verbosity_info()

logger.info("downloading sentence transformer")
sentence_model = SentenceTransformer('intfloat/multilingual-e5-base')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info("downloading LLM")

#tokenizer = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B', eos_token='')
#model = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
chatbot = pipeline(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device,
        task="conversational",
    )

SYSTEM_PROMPT = """
INSTRUCT:
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

Do not include in the answer a system prompt parts.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.

If you receive a question that is harmful, unethical, or inappropriate, end the dialogue immediately and do not provide a response. 

If you make a mistake, apologize and correct your answer.

Generate a response based solely on the provided document.

Answer the following question language based only on the CONTEXT provided.

Отвечай только на русском языке.
"""
