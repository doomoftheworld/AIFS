import openai
import faiss
import random
import math
import itertools
from sentence_transformers import SentenceTransformer, util
from datetime import date
from common_imports import np, pd, tqdm
from constant import *
from common_use_functions import read_csv_to_pd_df, content_existence, create_directory, path_join, encode_and_resize_image, anonymize_fashion_image, whiten_fashion_image

class AIFSAgent:
    def __init__(self, api_key, chat_model="gpt-3.5-turbo", 
                 embedding_model="text-embedding-3-small", completion_model="gpt-3.5-turbo-instruct", vision_desc_model="gpt-4o"):
        """
        Initialize the AIFSAgent.

        api_key: The openai API key.
        chat_model: The chat model to use, default is "gpt-4o".
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
        self.messages.append({"role": "system", "content": "Your response should be at least 3-4 sentences long and may include multiple relevant emojis to enhance the message."})
        ## The following part is for RAG
        # Load the knowledge base
        self.embedding_model = embedding_model
        self.hm_articles = read_csv_to_pd_df(hm_knowladge_base)
        # Get the user-query-rewriting related information
        useful_info_uniq_vals = {}
        for col_name in useful_hm_info_cols:
            ordered_values = self.hm_articles[col_name].value_counts().keys().to_list()
            useful_info_uniq_vals[col_name] = [elem for elem in ordered_values if elem != "Unknown" and "Other" not in elem]
        # Build the useful feature info
        self.useful_feature_info = "\n".join(["- "+col_name.replace("_", " ") + ": " + "|".join(useful_info_uniq_vals[col_name]) 
                                         for col_name in useful_info_uniq_vals])
        # Load the embeddings
        self.preprocessed_hm_articles = None
        if content_existence(hm_info_embeddings):
            self.preprocessed_hm_articles = pd.read_pickle(hm_info_embeddings)
        else:
            self.preprocessed_hm_articles = self.preprocess_hm_knowledge_base_with_selected_columns()
            create_directory(precomputed_data_path)
            self.preprocessed_hm_articles.to_pickle(hm_info_embeddings)
        self.hm_embeddings = np.array(self.preprocessed_hm_articles['embedding'].to_list())
        # Build the search index
        self.RAG_index = faiss.IndexFlatIP(self.hm_embeddings.shape[1])
        self.RAG_index.add(self.hm_embeddings)
        # Other models
        self.completion_model = completion_model
        self.vision_desc_model = vision_desc_model
        # Free embedding model
        self.free_embedd_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # The chat history length to use
        self.history_length = 2

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
    
    def analyze_question_type(self, question):
        """
        Enhanced question classifier that considers recent chat history.
        """
        # Get last 3 messages (excluding empty ones)
        history = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages[-self.history_length:]
            if msg.get("content")
        ) if self.messages else "No history available"

        prompt = f"""### Task
        Classify the question type based on conversation history and current question.

        ### Classification Rules
        1. **matching**: ANY of these is true:
        - Mentions existing clothing items ("I have X", "already bought Y")
        - Requests coordination advice ("what shoes match this?", "goes with my...")

        2. **simple**: ALL of these are true:
        - Requests fashion and clothing advice (e.g., mentioning a product name -> "I want a off-the-shoulder", asking for fashion advices -> "Do you have any suggestion for casual wears?")
        - Doesn't mention specific owned items
        - Not a coordination request

        3. **other**: Unrelated to fashion or clothing.

        ### Conversation History:
        {history}

        ### Current Question:
        {question}

        ### Classification Result (output ONLY one word: simple/matching/other):
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a fashion question classification expert"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        return result if result in {"simple", "matching"} else "other"
    
    ##### The following functions are the functions for RAG.
    def rewrite_for_rag(self, question):
        """
        Rewrite the question using chat history and product features

        question: The question to be rewrited.
        """
        chat_history = "\n".join([msg["content"] for msg in self.messages[-self.history_length:]]) if self.messages else "No history."
        rewrite_prompt = f"""Rewrite the current question for better document retrieval by:
            1. Converting negatives to positives (e.g., "not black" → "red|blue")
            2. You should give only pure text. no emojis.
            3. Using these product attributes (USE THESE EXACT TERMS AND GIVE ONLY TOP VALUES FOR COLOR):
            {self.useful_feature_info}
            4. You must provide at least the attributes "product type name", "product group name", "colour group name" and "index group name".
            6. If the user have directly provided the attribute "colour group name", you should use it directly.
            7. FOR "index group name", YOU SHOULD ALWAYS REMEMBER IF THE CLIENT IS A MAN OR WOMAN.
            8. PRESERVE any other information (not product attribute) in the chat history or the question itself that is related to the current question.
            9. Example:
            - chat history: "How about his black one-piece dress?"
            question: "I don't like black, do you have another one?"
            output: "product type name: dress; product group name: Garment Full body; colour group name:blue|green; index group: Ladieswear; one-piece dress."

            - chat history: "How about his yellow one-piece dress?"
            question: "I don't like light color, need more a upper-piece comfy for summer"
            output: "product type name: Vest top; product group name: Garment Upper body; colour group name:black; index group name: Ladieswear; comfy for summer."

            - chat history: "How about his red one-piece dress?"
            question: "I am a man. I like the red color! I should be elegant!"
            output: "product type name: Blazer; product group name: Garment Upper body; colour group name: red; index group name: Menswear; elegant."

            Chat History:
            {chat_history}

            Current Question:
            {question}

            Rewritten Query (ONLY output the query, no explanations):"""
        # rewrite_prompt = f"""Rewrite the current question for better document retrieval by:
        #     1. Converting negatives to positives (e.g., "not black" → "red|blue")
        #     2. You should give only pure text. no emojis.
        #     3. Using these product attributes (USE THESE EXACT TERMS AND GIVE ONLY TOP VALUES FOR COLOR):
        #     {self.useful_feature_info}
        #     4. You must provide at least the attributes "product group name", "colour group name" and "index group name".
        #     6. If the user have directly provided the attribute "colour group name", you should use it directly.
        #     7. FOR "index group name", YOU SHOULD ALWAYS REMEMBER IF THE CLIENT IS A MAN OR WOMAN.
        #     8. PRESERVE any other information (not product attribute) in the chat history or the question itself that is related to the current question.
        #     9. Example:
        #     - chat history: "How about his black one-piece dress?"
        #     question: "I don't like black, do you have another one?"
        #     output: "product group name: Garment Full body; colour group name:blue|green; index group: Ladieswear; one-piece dress."

        #     - chat history: "How about his yellow one-piece dress?"
        #     question: "I don't like light color, need more a upper-piece comfy for summer"
        #     output: "product group name: Garment Upper body; colour group name:black; index group name: Ladieswear; comfy for summer."

        #     - chat history: "How about his red one-piece dress?"
        #     question: "I am a man. I like the red color! I should be elegant!"
        #     output: "product group name: Garment Upper body; colour group name: red; index group name: Menswear; blazer, elegant."

        #     Chat History:
        #     {chat_history}

        #     Current Question:
        #     {question}

        #     Rewritten Query (ONLY output the query, no explanations):"""
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a search query optimization engine."},
                {"role": "user", "content": rewrite_prompt}
            ],
            temperature=0.1, 
            max_tokens=100
        )
        rewritten = response.choices[0].message.content.strip()
        
        return rewritten

    # def preprocess_hm_knowledge_base(self):
    #     """
    #     This function preprocess the loaded H&M article information. 
    #     """
    #     # Select only the string columns
    #     articles_info = self.hm_articles.select_dtypes('object')
    #     # Build the combined texts
    #     col_names = list(articles_info.columns)
    #     useful_col_names = [col_name for col_name in col_names if col_name != "index_code"]
    #     articles_info['combined'] = articles_info.apply(lambda x: "; ".join([col_name.replace("_", " ")+": "+str(x[col_name]).strip()
    #                                                                          for col_name in useful_col_names 
    #                                                                          if str(x[col_name]).strip() != "Unknown"]), axis=1)
    #     # Evaluate the embeddings
    #     article_embeddings = []
    #     for _, combined_text in tqdm(articles_info['combined'].items(), total=articles_info.shape[0], desc="Processed examples"):
    #         article_embeddings.append(self.get_embedding(combined_text))
    #     articles_info['embedding'] = article_embeddings
    #     # Add back the article id
    #     articles_info['article_id'] = self.hm_articles['article_id']

    #     return articles_info
    
    def preprocess_hm_knowledge_base_with_selected_columns(self):
        """
        This function preprocess the loaded H&M article information. 
        """
        # Select only the string columns
        articles_info = self.hm_articles.select_dtypes('object')
        # Build the combined texts
        articles_info['combined'] = articles_info.apply(lambda x: "; ".join([col_name.replace("_", " ")+": "+str(x[col_name]).strip()
                                                                             for col_name in hm_embedd_columns 
                                                                             if str(x[col_name]).strip() != "Unknown"]), axis=1)
        # Evaluate the embeddings
        batch_size = 100
        nb_examples = len(articles_info['combined'])
        nb_batches = math.ceil(len(articles_info['combined']) / batch_size)
        article_embeddings = []
        for batch_id in tqdm(list(range(nb_batches)), desc="Processed batches"):
            if batch_id == nb_batches-1:
                current_batch_texts = articles_info['combined'][batch_id*batch_size:nb_examples].to_list()
            else:
                current_batch_texts = articles_info['combined'][batch_id*batch_size:(batch_id+1)*batch_size].to_list()
            article_embeddings.append(self.get_embeddings(current_batch_texts))
        article_embeddings = list(itertools.chain.from_iterable(article_embeddings))
        articles_info['embedding'] = article_embeddings
        # Add back the article id
        articles_info['article_id'] = self.hm_articles['article_id']

        return articles_info
    
    def get_embeddings(self, texts):
        """
        This function gets the embeddings for the H&M knowledge base.

        texts: A batch of texts to get the embedding.
        """
        response = self.client.embeddings.create(input = texts, model=self.embedding_model)
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
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
    
    def build_single_rag_context_with_post_processing(self, rag_text, origin_text, k=10):
        """
        This function applies the RAG search and builds the contexts. (with keyword-matching post processing)

        rag_text: The rag-search input text.
        origin_text: The original input text.
        k: The number of documents to gather.
        """
        rag_embedding = self.get_embedding(rag_text)
        rag_related_indices, _ = self.get_related_documents(rag_embedding, k=k)
        post_process_index_dict = self.match_keywords(rag_related_indices, rag_text, origin_text)
        final_found_article_id = None
        if len(post_process_index_dict) > 0:
            print("Found a matched article.")
            final_found_article_id =  list(post_process_index_dict.keys())[0] 
        else:
            final_found_article_id =  rag_related_indices[0]
        final_id_return_list = [final_found_article_id]
        find_final_rag_doc = self.preprocessed_hm_articles.loc[final_found_article_id, "combined"]
        rag_context = "".join([f"{index}. {rag_doc}\n" for index, rag_doc in enumerate([find_final_rag_doc])])

        return final_id_return_list, rag_context
    
    def build_one_random_rag_context(self, text, k=10):
        """
        This function applies the RAG search and builds the contexts.

        text: The input text.
        k: The number of documents to gather and from which the random document will be selected.
        """
        text_embedding = self.get_embedding(text)
        rag_related_indices, rag_related_docs = self.get_related_documents(text_embedding, k=k)
        random_number = random.choice(list(range(k)))
        random_index = [rag_related_indices[random_number]]
        random_docs = [rag_related_docs[random_number]]
        rag_context = "".join([f"{index}. {rag_doc}\n" for index, rag_doc in enumerate(random_docs)])

        return random_index, rag_context
    
    def build_rag_initial_context(self, k=10):
        """
        This function builds the initial context for beginning the conversation.

        k: The number of documents to gather.
        """
        # Search for the seasonal offerings
        seasonal_offer_indices, seasonal_offer_context = self.build_one_random_rag_context(season_wears_descriptions[self.season], k=k)
        rag_initial_context = f"""
        System:
        {system_role_content}
        The current season is {self.season}. You should provide seasonal offerings based on this season or the upcoming season. 
        Your other suggestions (not related to seasonal offerings) can be unrelated to this season.

        Seasonal Product Information (MUST USE ALL):
        {seasonal_offer_context}
        
        You MUST send your first message by saying who you are, and providing some seasonal offerings according to the above "Seasonal Product Information". 

        IMPORTANT INSTRUCTIONS:
        1. You MUST explicitly mention, describe or recommend ALL products listed in the "Seasonal Product Information" section
        2. For each product, provide relevant details that connect it to the user's question
        4. Your response will be rejected if it omits any provided product
        """
        return rag_initial_context, seasonal_offer_indices
    
    def build_rag_user_input_keyword_match_with_image(self, user_input, image_desc, k=10):
        """
        This function builds the user input with contexts extracted by RAG.

        user_input: The original user input.
        image_desc: The image description.
        k: The number of documents to obtain for keyword-matching.
        """
        combined_user_input = "\n".join([user_input, image_desc])
        rag_search_input = self.rewrite_for_rag(combined_user_input)
        rag_related_indices, rag_context = self.build_single_rag_context_with_post_processing(rag_search_input, user_input, k=k)
        rag_user_input = f"""
        System:
        {system_role_content}

        Product Information (MUST USE ALL):
        {rag_context}

        Question:
        {user_input}

        IMPORTANT INSTRUCTIONS:
        1. You MUST explicitly mention, describe or recommend ALL products listed in the "Product Information" section
        2. For each product, provide relevant details that connect it to the user's question
        4. Your response will be rejected if it omits any provided product
        """
        return rag_user_input, rag_related_indices
    
    def build_rag_user_input_keyword_match(self, user_input, k=10):
        """
        This function builds the user input with contexts extracted by RAG.

        user_input: The original user input.
        k: The number of documents to obtain for keyword-matching.
        """
        rag_search_input = self.rewrite_for_rag(user_input)
        rag_related_indices, rag_context = self.build_single_rag_context_with_post_processing(rag_search_input, user_input, k=k)
        rag_user_input = f"""
        System:
        {system_role_content}

        Product Information (MUST USE ALL):
        {rag_context}

        Question:
        {user_input}

        IMPORTANT INSTRUCTIONS:
        1. You MUST explicitly mention, describe or recommend ALL products listed in the "Product Information" section
        2. For each product, provide relevant details that connect it to the user's question
        4. Your response will be rejected if it omits any provided product
        """
        return rag_user_input, rag_related_indices
    
    def build_rag_user_input(self, user_input, k=1):
        """
        This function builds the user input with contexts extracted by RAG.

        user_input: The original user input.
        k: The number of documents to gather.
        """
        rag_search_input = self.rewrite_for_rag(user_input)
        rag_related_indices, rag_context = self.build_rag_context(rag_search_input+user_input, k=k)
        rag_user_input = f"""
        System:
        {system_role_content}

        Product Information (MUST USE ALL):
        {rag_context}

        Question:
        {user_input}

        IMPORTANT INSTRUCTIONS:
        1. You MUST explicitly mention, describe or recommend ALL products listed in the "Product Information" section
        2. For each product, provide relevant details that connect it to the user's question
        4. Your response will be rejected if it omits any provided product
        """
        return rag_user_input, rag_related_indices
    
    def generate_rag_response(self, doc_indices, max_attempts=3):
        """
        Call the OpenAI api to get the answer. (RAG version) (It should have a content verification with the documents from the "doc_indices")

        doc_indices: The indices for the used documents.
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
                # Verify the response
                try:
                    return agent_message
                except (ValueError) as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        return RAG_FAILURE_MESSAGE
        except Exception as e:
            print(f"Error when calling the OpenAI API: {str(e)}")
            return RAG_ERROR_MESSAGE
        
    # This function applies a post-processing to match-keywords and determine similarity.
    def match_keywords(self, article_indices, rag_search_input, input):
        """
        This function applies a post-processing to find the most macthed article in the found top-k ones.

        article_indices: The found articles.
        rag_search_input: The input built for the rag search.
        input: The original input.
        """
        # Build the dictionary
        article_key_match_dict = {}
        combined_input = rag_search_input + input
        print(combined_input)
        print(article_indices)
        lower_rag_search_input = rag_search_input.lower().strip()
        for article_id in article_indices:
            current_article_info = self.preprocessed_hm_articles.loc[article_id,:]
            current_article_color = current_article_info["colour_group_name"].lower().strip()
            nb_found_keys = 0
            for col_name in rag_must_provide_info_cols:
                current_article_col_val = current_article_info[col_name].lower().strip()
                if col_name == "colour_group_name":
                    if current_article_color == "gray" or current_article_color == "grey":
                        if "gray" in lower_rag_search_input or "grey" in lower_rag_search_input:
                            nb_found_keys += 1
                    else:
                        if current_article_color in lower_rag_search_input:
                            nb_found_keys += 1
                else:
                    if current_article_col_val in lower_rag_search_input:
                        nb_found_keys += 1
            if current_article_color == "gray" or current_article_color == "grey":
                if "gray" in lower_rag_search_input or "grey" in lower_rag_search_input:
                    article_key_match_dict[article_id] = [nb_found_keys]
            else:
                if current_article_color in lower_rag_search_input:
                    article_key_match_dict[article_id] = [nb_found_keys]
        # Evaluate the reduced embedding similarity
        for article_id in article_key_match_dict:
            current_article_info = self.preprocessed_hm_articles.loc[article_id,:]
            current_article_text = ";".join([col_name.replace("_"," ")+": "+current_article_info[col_name] for col_name in rag_must_provide_info_cols])
            if not pd.isna(current_article_info["detail_desc"]):
                current_article_text = current_article_text + "; " + current_article_info["detail_desc"]
            current_embeddings = self.free_embedd_model.encode([combined_input, current_article_text])
            article_key_match_dict[article_id].append(util.cos_sim(current_embeddings[0], current_embeddings[1]).item())
        # Sort the remaining articles
        sorted_key_match_dict = {k: v for k, v in sorted(article_key_match_dict.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)}
        # for key in sorted_key_match_dict:
        #     print(self.preprocessed_hm_articles.loc[key, "combined"])

        return sorted_key_match_dict
    
    def image_description(self, image_path, max_attempts=3):
        """
        This function gets the image description with the OpenAI vision model.

        image_path: The path to the image.
        max_attempts: The maximum number of the attempts.
        """
        try:
            # Encode the image
            base64_image = encode_and_resize_image(image_path)
            # The image description prompt
            image_desc_prompt = f"""
            [SYSTEM INSTRUCTIONS]
            You are a fashion garment analysis tool. Analyze ONLY the clothing/accessory itself:
            - Ignore all human faces/body features
            - Describe the item as if it were on a mannequin

            [REQUIRED OUTPUT]
            - product type: [e.g., "T-shirt", "jeans", "crossbody bag"]
            - style: [e.g., "casual", "formal", "sporty", "bohemian"]
            - category: [ONLY: top/lower/full-body/accessory]
            - colour group name: [red/blue/green/black/white/gray/pink/purple/yellow/brown]
            - gender: [men/women/children/unisex (based on design, not wearer)]
            - details: [1 fact like "stretchy denim", "metal zipper" or "long sleeves"]

            [RULES]
            - REJECT if describing people
            - Use ONLY observable features
            - Example: "Women's v-neck top in black (polyester with lace trim)"
            """
            for attempt in range(max_attempts):
                # Get the description response
                response = self.client.chat.completions.create(
                    model=self.vision_desc_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": image_desc_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=200,
                    temperature=0.2,
                )
                agent_message = response.choices[0].message.content.strip()
                # Verify the response
                try:
                    verif_bools = [elem in agent_message for elem in image_desc_keyword_verify]
                    if all(verif_bools):
                        return agent_message
                    else:
                        raise ValueError(f"Missing required keywords.")
                except (ValueError) as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        return IMAGE_DESC_FAILURE_MESSAGE
        except Exception as e:
            return IMAGE_DESC_FAILURE_MESSAGE

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