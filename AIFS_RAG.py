# import openai
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# class AIFSRAG:
#     def __init__(self, openai_api_key, embedding_model="text-embedding-ada-002", generation_model="gpt-3.5-turbo"):
#         """
#         Initialize the RAG class with OpenAI API key and model names.
        
#         :param openai_api_key: Your OpenAI API key.
#         :param embedding_model: The name of the embedding model to use (default: "text-embedding-ada-002").
#         :param generation_model: The name of the generation model to use (default: "gpt-3.5-turbo").
#         """
#         openai.api_key = openai_api_key
#         self.embedding_model = embedding_model
#         self.generation_model = generation_model
#         self.documents = []
#         self.document_embeddings = []

#     def add_document(self, document):
#         """
#         Add a document to the knowledge base and generate its embedding.
        
#         :param document: The document text to add.
#         """
#         self.documents.append(document)
#         embedding = self._get_embedding(document)
#         self.document_embeddings.append(embedding)

#     def _get_embedding(self, text):
#         """
#         Generate an embedding for the given text using OpenAI's embedding model.
        
#         :param text: The text to generate an embedding for.
#         :return: The embedding vector.
#         """
#         response = openai.Embedding.create(
#             input=text,
#             model=self.embedding_model
#         )
#         return response['data'][0]['embedding']

#     def _find_most_relevant_document(self, query_embedding):
#         """
#         Find the most relevant document in the knowledge base for the given query embedding.
        
#         :param query_embedding: The embedding of the query.
#         :return: The most relevant document text.
#         """
#         similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
#         most_relevant_index = np.argmax(similarities)
#         return self.documents[most_relevant_index]

#     def ask(self, query):
#         """
#         Ask a question and get an answer using the RAG model.
        
#         :param query: The question to ask.
#         :return: The generated answer.
#         """
#         query_embedding = self._get_embedding(query)
#         relevant_document = self._find_most_relevant_document(query_embedding)
        
#         prompt = f"Document: {relevant_document}\n\nQuestion: {query}\nAnswer:"
        
#         response = openai.Completion.create(
#             model=self.generation_model,
#             prompt=prompt,
#             max_tokens=150,
#             temperature=0.7
#         )
        
#         return response.choices[0].text.strip()