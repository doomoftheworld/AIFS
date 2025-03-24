import openai
import faiss
import json
from datetime import date
from common_imports import np, pd, tqdm
from constant import *
from common_use_functions import read_csv_to_pd_df, content_existence, create_directory, path_join

class AIFSAgent:
    def __init__(self, api_key, chat_model="gpt-3.5-turbo", embedding_model="text-embedding-3-small", completion_model="gpt-3.5-turbo-instruct"):
        """
        Initialize the AIFSAgent.

        api_key: The openai API key.
        chat_model: The chat model to use, default is "gpt-3.5-turbo".
        embedding_model: 
        """
        # Create the client with the API key
        self.client = openai.OpenAI(api_key=api_key)
        ## This part is for the chat configuration
        self.chat_model = chat_model
        self.messages = []
        # Get the current season
        self.season = self.get_season(date=date.today())
        # Add the system message to give the background
        # self.messages.append({"role": "system", "content": system_role_content})
        ## The following part is for RAG
        # Load the knowledge base
        self.embedding_model = embedding_model
        self.hm_articles = read_csv_to_pd_df(hm_knowladge_base)
        # Load the embeddings
        self.preprocessed_hm_articles = None
        if content_existence(hm_info_embeddings):
            self.preprocessed_hm_articles = pd.read_pickle(hm_info_embeddings)
        else:
            self.preprocessed_hm_articles = self.preprocess_hm_knowledge_base()
            create_directory(precomputed_data_path)
            self.preprocessed_hm_articles.to_pickle(hm_info_embeddings)
        self.hm_embeddings = np.array(self.preprocessed_hm_articles['embedding'].to_list())
        # Build the search index
        self.RAG_index = faiss.IndexFlatIP(self.hm_embeddings.shape[1])
        self.RAG_index.add(self.hm_embeddings)
        # Type analysis model
        self.completion_model = completion_model

    # Season determination
    def get_season(self, date):
        """
        This function is for the season determination based on the current date.

        date: The actual date.
        """
        m = date.month
        x = m%12 // 3 + 1
        if x == 1:
            season = "Winter"
        if x == 2:
            season = "Spring"
        if x == 3:
            season = "Summer"
        if x == 4:
            season = "Autumn"

        return season
    
    # Analyze the question type
    def analyze_question_type(self, question):
        """
        This function analyze if the question itself is a question for searching advices or just a simple question.

        question: The text of the question.
        """
        try:
            # Define the prompt for question type analysis
            type_analyze_prompt = f"""
            Please help me determine the type of the provided question, the type of question could be:
            1. simple: This question is about asking for simple style advices.
            2. matching: This question is about asking for style-matching advices based on the already bought clothes.
            3. other: This question is not related to any style recommendation.

            Examples:
            1. I would like to buy some clothes for the current season. -> simple
            2. I don't think I look good in jeans. -> simple
            3. I have already some jeans and boots, I would like to buy some other clothes. -> matching
            4. What do you think about the weather today -> other
            5. I have a pair of jeans, but I don't know what shoes to wear with them. -> matching
            6. What should I wear to a prom? -> simple

            Question: {question}

            Type: (Please give directly the type name.)
            """

            # Get the response
            response = self.client.completions.create(
                model=self.completion_model, 
                prompt=type_analyze_prompt,
                max_tokens=10,
                temperature=0.3
            )

            # Extract the question type
            question_type = response.choices[0].text.strip()

            # Validate the question type
            valid_types = {"simple", "matching", "other"}
            if question_type not in valid_types:
                raise ValueError(f"Invalid question type returned: {question_type}")

            return question_type
        except ValueError as e:
            # Log the error and return a default value
            print(f"Error: {e}. Defaulting to 'other'.")
            return "other"

    ##### The following functions are the functions for RAG.
    def preprocess_hm_knowledge_base(self):
        """
        This function preprocess the loaded H&M article information. 
        """
        # Select only the string columns
        articles_info = self.hm_articles.select_dtypes('object')
        # Build the combined texts
        col_names = list(articles_info.columns)
        useful_col_names = [col_name for col_name in col_names if col_name != "index_code"]
        articles_info['combined'] = articles_info.apply(lambda x: "; ".join([col_name.replace("_", " ")+": "+str(x[col_name]).strip()
                                                                             for col_name in useful_col_names 
                                                                             if str(x[col_name]).strip() != "Unknown"]), axis=1)
        # Evaluate the embeddings
        article_embeddings = []
        for _, combined_text in tqdm(articles_info['combined'].items(), total=articles_info.shape[0], desc="Processed examples"):
            article_embeddings.append(self.get_embedding(combined_text))
        articles_info['embedding'] = article_embeddings
        # Add back the article id
        articles_info['article_id'] = self.hm_articles['article_id']

        return articles_info
    
    def get_embedding(self, text):
        """
        This function gets the embeddings for the H&M knowledge base.

        text: The text for which we would like to have the embedding.
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=self.embedding_model).data[0].embedding
    
    def get_related_documents(self, text_embedding, k=3):
        """
        This function gets the top-k related documents from the H&M knowledge base.

        text_embedding: The text embedding for one sentence.
        k: The number indicating the rank for the top-k.
        """
        D, I = self.RAG_index.search(np.array([text_embedding]), k)
        found_index = I[0]
        found_context = self.preprocessed_hm_articles.loc[found_index, "combined"].to_list()
        
        return found_index, found_context
    
    def get_image_path(self, article_index):
        """
        This function finds the path of the article image to be displayed.

        article_index: The index of the article in the knowledge base.

        Note: This function process only one image.
        """
        article_id = str(self.preprocessed_hm_articles.loc[article_index, "article_id"])
        article_image_path = path_join(hm_image_folder, "0"+article_id[:2], "0"+article_id+jpg_ext)

        return article_image_path
    
    def build_rag_context(self, text, k=1):
        """
        This function applies the RAG search and builds the contexts.

        text: The input text.
        k: The number of documents to gather.
        """
        text_embedding = self.get_embedding(text)
        rag_related_indices, rag_related_docs = self.get_related_documents(text_embedding, k=k)
        rag_context = "".join([f"{index}. {rag_doc}\n" for index, rag_doc in enumerate(rag_related_docs)])

        return rag_related_indices, rag_context
    
    def build_rag_initial_context(self, k=1):
        """
        This function builds the initial context for beginning the conversation.

        k: The number of documents to gather.
        """
        # Search for the seasonal offerings
        seasonal_offer_indices, seasonal_offer_context = self.build_rag_context(season_wears_descriptions[self.season], k=k)
        rag_initial_context = f"""
        System:
        {system_role_content}
        The current season is {self.season}. You should provide seasonal offerings based on this season. 
        Your other suggestions (not related to seasonal offerings) can be unrelated to this season.

        Context:
        {seasonal_offer_context}
        
        You MUST begin the conversation by saying who you are. 
        Your response should be at least 3-4 sentences long and may include multiple relevant emojis to enhance the message. 
        You should use the product information from the above "Context".

        IMPORTANT: Your response MUST be a valid JSON object. If it is not in JSON format, it will be rejected.
        
        Your first message should be a valid JSON object with the following keys:
        - "answer": Your greeting message and some seasonal offerings. DO NOT include context numbers in the message. Ensure it directly references or summarizes specific details from the context.
        - "used_context": A list of context numbers you **directly used** to generate the answer. Only include context numbers if you explicitly referenced or derived information from them.

        After generating your response, please verify that it is a valid JSON object. If it is not, correct the format before submitting.

        Example:
        {{
          "answer": "Your message here.",
          "used_context": [0, 1]
        }}
        """
        return rag_initial_context, seasonal_offer_indices
    
    def build_rag_user_input(self, user_input, k=1):
        """
        This function builds the user input with contexts extracted by RAG.

        user_input: The original user input.
        k: The number of documents to gather.
        """
        rag_related_indices, rag_context = self.build_rag_context(user_input, k=k)
        rag_user_input = f"""
        System:
        {system_role_content}
        The current season is {self.season}. You should provide seasonal offerings based on this season. 
        Your other suggestions (not related to seasonal offerings) can be unrelated to this season.

        Context:
        {rag_context}

        Question:
        {user_input}

        IMPORTANT: Your response MUST be a valid JSON object. If it is not in JSON format, it will be rejected.

        Please answer the question with the product information in the above "Context" section.
        Your response should be at least 3-4 sentences long and may include multiple relevant emojis to enhance the message. 
        
        Your response should be a valid JSON object with the following keys:
        - "answer": A string containing your answer. DO NOT include context numbers in the answer. Ensure it directly references or summarizes specific details from the context.
        - "used_context": A list of context numbers you **directly used** to generate the answer. Only include context numbers if you explicitly referenced or derived information from them.

        After generating your response, please verify that it is a valid JSON object. If it is not, correct the format before submitting.
        
        Example:
        {{
          "answer": "Your answer here.",
          "used_context": [0, 1]
        }}
        """
        return rag_user_input, rag_related_indices
    
    def generate_rag_response(self, nb_context, max_attempts=3):
        """
        Call the OpenAI api to get the answer. (RAG version)

        nb_context: The number of provided context.
        max_attempts : Maximum number of retry attempts.
        """
        try:
            for attempt in range(max_attempts):
                # Generate the response
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=self.messages
                )
                agent_message = response.choices[0].message.content.strip() # Avoid the "not subscriptable" error
                print(agent_message)
                # Verify the response
                try:
                    agent_message_data = json.loads(agent_message)
                    if "answer" not in agent_message_data or "used_context" not in agent_message_data:
                        raise ValueError("Invalid response: missing 'answer' or 'used_context'")
                    if not isinstance(agent_message_data["used_context"], list):
                        raise ValueError("Invalid response: 'used_context' should be a list")
                    if agent_message_data["used_context"] and max(agent_message_data["used_context"]) > nb_context-1:
                        raise ValueError("Invalid response: 'used_context' have invalid context ids")
                    return agent_message_data
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        return RAG_FAILURE_MESSAGE
        except Exception as e:
            print(f"Error when calling the OpenAI API: {str(e)}")
            return RAG_ERROR_MESSAGE


    ##### The following functions are the functions for chatting.
    def add_message(self, role, content):
        """
        Add a message.

        role: The role of the message (system, user or assitant).
        content: The message.
        """
        self.messages.append({"role": role, "content": content})

    def remove_last_message(self):
        """
        Remove the last message.
        """
        if self.messages:
            self.messages.pop()

    def get_history(self):
        """
        Get the whole message history.
        """
        return self.messages

    def clear_history(self):
        """
        Clear all the message history.
        """
        self.messages = []

    def generate_response(self):
        """
        Call the OpenAI api to get the answer.
        """
        try:
            # Generate the response
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=self.messages
            )
            agent_message = response.choices[0].message.content.strip() # Avoid the "not subscriptable" error
            return agent_message
        except Exception as e:
            print(f"Error when calling the OpenAI API: {str(e)}")
            return "Sorry, I've encountered some internal errors, please ask again."