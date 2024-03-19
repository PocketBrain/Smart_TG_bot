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
    result_documents = result_documents[:5]

    return result_documents
