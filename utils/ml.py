def remove_after_question(text):
    question_pos = text.find("Вопрос :")
    if question_pos != -1:
        return text[:question_pos]
    else:
        return text


def encodeQuestion(question, model, collection):
    embedding = model.encode(question, normalize_embeddings=True)
    embedding_list = embedding.tolist()
    result = collection.query(query_embeddings=[embedding_list])
    result_documents = []

    for distance, metadata, document in zip(
            result["distances"][0], result["metadatas"][0], result["documents"][0]
    ):
        if distance < 1:
            result_documents.append({"file_path": metadata["file_path"], "answer": document, "metric": distance})
    result_documents.sort(key=lambda x: x["metric"], reverse=False)
    result_documents = result_documents[:3]

    return result_documents


qa_template = """
        <SC1>Вы - банковский ассистент. Вы отвечаете на вопрос с помощью информации,вы можете использовать ее при ответе если вопрос.
        Отвечай только на то, о чем тебя спросили. Не задавай встречных вопросов.
        Не повторяй информацию, только отвечай на вопрос. Давай один точный ответ только по информации. Не повторяй вопрос
        Информация: {context}
        Вопрос: {question}
        Ответ: <extra_id_0>
        """
