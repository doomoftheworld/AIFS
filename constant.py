"""
The configurable parameters for the application.
"""
max_image_size = (200, 200)
jpg_ext = ".jpg"
used_chat_model = "gpt-3.5-turbo"
used_embedding_model = "text-embedding-3-small"
used_completion_model = "gpt-3.5-turbo-instruct"
precomputed_data_path = "precomputed_data"
hm_image_folder = "h-and-m-personalized-fashion-recommendations\\images\\"
hm_knowladge_base = "h-and-m-personalized-fashion-recommendations\\articles.csv"
hm_info_embeddings = precomputed_data_path+"\\hm_info_embeddings.pkl"
system_role_content = "You are an AI Fashion Stylist who knows H&M products very well. You interact proactively with customers to give recommendations based on their preferences and information gathered from the H&M inventory. You should sound very polite and enthusiastic during the conversation."
season_wears_descriptions = {
    "Winter":"Warm and cozy winter apparel designed for cold climates, featuring coats, sweaters, and thermal layers.",
    "Spring":"Fresh and vibrant spring styles, ideal for mild weather, including light jackets, floral dresses, and casual pants.",
    "Summer":"Lightweight and breathable summer clothing, perfect for hot weather, including shorts, dresses, and tank tops.",
    "Autumn":"Versatile autumn wear for transitional weather, with options like jackets, scarves, and long-sleeve tops.",
}
useful_hm_info_cols = ["product_group_name", "graphical_appearance_name", "colour_group_name", "index_group_name"]
rag_must_provide_info_cols = ["product_group_name", "colour_group_name", "index_group_name"]
RAG_FAILURE_MESSAGE = "Sorry, I failed to answer your question, please ask again."
RAG_ERROR_MESSAGE = "Sorry, I've encountered some internal errors, please ask again."