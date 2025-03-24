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
# inter_context_prevent_content = "Please answer the question based on the provided context. Ignore any previous context unless explicitly mentioned."
season_wears_descriptions = {
    "Winter":"Warm and cozy winter apparel designed for cold climates, featuring coats, sweaters, and thermal layers.",
    "Spring":"Fresh and vibrant spring styles, ideal for mild weather, including light jackets, floral dresses, and casual pants.",
    "Summer":"Lightweight and breathable summer clothing, perfect for hot weather, including shorts, dresses, and tank tops.",
    "Autumn":"Versatile autumn wear for transitional weather, with options like jackets, scarves, and long-sleeve tops.",
}
RAG_FAILURE_MESSAGE = {
    "answer": "Sorry, I failed to answer your question, please ask again.",
    "used_context": []
}
RAG_ERROR_MESSAGE = {
    "answer": "Sorry, I've encountered some internal errors, please ask again.",
    "used_context": []
}