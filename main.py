from loguru import logger
from telegram.ext import Updater, CommandHandler, MessageHandler, filters, Filters
from telegram import ReplyKeyboardMarkup

import torch

from config.chromadb_client import collection_result
from config.ml import sentence_model, tokenizer, device, model
from utils.ml import qa_template, encodeQuestion, remove_after_question

# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)',
#     level=logging.DEBUG)

Telegram_token = '7189109603:AAFF0hyVp8yH_8Cy-cuQGEaShaVLkN9afOE'

def wake_up(update, context):
    chat = update.effective_chat
    name = update.message.chat.first_name
    button = ReplyKeyboardMarkup([['Узнать курс валют', 'Спросить о чем-то']], resize_keyboard=True)
    context.bot.send_message(chat_id=chat.id,
                             text='Привет, {}. Этот чат бот ЦБ РФ поможет тебе по всем интересующим тебя вопросам!'.format(
                                 name),
                             reply_markup=button)


def LLMChat(update, context):
    user_input = update.message.text
    top_answer = encodeQuestion(user_input, sentence_model, collection_result)
    result = []
    for answer in top_answer:
        result.append(answer['answer'])

    if len(top_answer) == 0:
        qa = qa_template.format(context="", question=user_input)
    else:
        qa = qa_template.format(context=result, question=user_input)

    input_ids = torch.tensor([tokenizer.encode(qa, add_special_tokens=True)]).to(device)

    outputs = model.generate(input_ids,
                             top_p=0.4,
                             temperature=0.2,
                             repetition_penalty=2.0,
                             min_length=20,
                             max_length=600,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=True)
    answer = remove_after_question(tokenizer.decode(outputs[0][1:]))

    response = f'Ваш ответ на вопрос: {user_input} - {answer}'
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def main():
    logger.info("starting telegram bot")
    updater = Updater(token=Telegram_token)
    updater.dispatcher.add_handler(CommandHandler('start', wake_up))
    updater.dispatcher.add_handler(MessageHandler(callback=LLMChat, filters=Filters.text & ~Filters.command))
    updater.start_polling(poll_interval=10.0)
    updater.idle()


if __name__ == '__main__':
    main()
