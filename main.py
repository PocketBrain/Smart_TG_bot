from loguru import logger
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, Filters
from telegram import ReplyKeyboardMarkup
from transformers import  Conversation
import torch
from transformers import pipeline
from config.chromadb_client import collection_result
from config.ml import sentence_model, chatbot, SYSTEM_PROMPT
from utils.ml import encodeQuestion


Telegram_token = '7189109603:AAFF0hyVp8yH_8Cy-cuQGEaShaVLkN9afOE'

def wake_up(update, context):
    chat = update.effective_chat
    name = update.message.chat.first_name
   # button = ReplyKeyboardMarkup([['Узнать курс валют', 'Спросить о чем-то']], resize_keyboard=True)
    context.bot.send_message(chat_id=chat.id,
                             text='Привет, {}. Этот чат мединского бота, задай мне вопрос!'.format(
                                 name),)
                             #reply_markup=button)

def generate_answer(
    prompt,
    conversation: Conversation,
    max_new_tokens: int = 80,
    temperature=0.5,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 2.0,
    do_sample: bool = True,
    num_beams: int = 2,
    early_stopping: bool = True,
) -> str:
    # Генерируем ответ от чатбота
    output = chatbot(
        conversation,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )
    print(output)
    # Возвращаем последнее сообщение чатбота как ответ
    return output
def LLMChat(update, context):
    conversation = Conversation()
    user_input = update.message.text
    top_answer = encodeQuestion(user_input, sentence_model, collection_result)
    result = []
    for answer in top_answer:
        result.append(answer['answer'])

    conversation.add_message({"role": "system", "content": SYSTEM_PROMPT})
    document_template = f"""
        CONTEXT:
        {result}
        Отвечай только на русском языке.
        ВОПРОС:{user_input}
        """
    conversation.add_message({"role": "user", "content": document_template})
    #prompt = chatbot.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    output = generate_answer(chatbot, conversation, temperature=0.5)
    print(output[-1]["content"])
    context.bot.send_message(chat_id=update.effective_chat.id, text=output[-1]["content"])


def main():
    logger.info("starting telegram bot")
    updater = Updater(token=Telegram_token)
    updater.dispatcher.add_handler(CommandHandler('start', wake_up))
    updater.dispatcher.add_handler(MessageHandler(callback=LLMChat, filters=Filters.text & ~Filters.command))
    updater.start_polling(poll_interval=10.0)
    updater.idle()


if __name__ == '__main__':
    main()
