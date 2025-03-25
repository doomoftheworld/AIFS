import tkinter as tk
import threading
from tkinter import scrolledtext, Text
from AIFS_agent import AIFSAgent
from constant import used_chat_model, used_embedding_model, used_completion_model, max_image_size
from common_use_functions import content_existence
from common_imports import Image, ImageTk

"""
The chat application
"""
class AIFSChatApp:
    def __init__(self, root, api_key):
        # Set the converssation window
        self.root = root
        self.root.title("AI Fashion Stylist")

        # The dialog display
        self.dialog_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI Emoji", 12))
        self.dialog_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # The user input
        self.text_entry = Text(root, height=5, wrap=tk.WORD)
        self.text_entry.pack(padx=10, pady=10, fill=tk.X)
        self.text_entry.bind("<Return>", self.handle_user_input)

        # The button frame that contains all the buttons.
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.RIGHT, pady=5, fill=tk.X)

        # The file upload button
        self.upload_button = tk.Button(self.button_frame, text="Image Upload", command=self.handle_image_upload)
        self.upload_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # The send button
        self.send_button = tk.Button(self.button_frame, text="Send", command=self.handle_user_input)
        self.send_button.pack(padx=10, side=tk.RIGHT, expand=True, fill=tk.X)

        # The AIFS agent to get the answer
        self.agent = AIFSAgent(api_key=api_key, chat_model=used_chat_model, embedding_model=used_embedding_model, completion_model=used_completion_model)

        # The attribute to keep the images (not recycled by the garbage system)
        self.image_references = []

        # The numbers for the RAG answering
        self.initial_greeting_k = 10
        self.rag_k = 1 # General version, it would be the real number of the gathered documents
        self.rag_k_keyword = 10 # for the keyword-matching version, the number of documents to gather for keyword-matching, it will output only one doc.

        # Display the intial information from the assistant
        # self.initial_greeting()

    def initial_greeting(self):
        """
        This function initialize the conversation with a response from the assistant.
        """
        # Block the send button in the first place
        self.text_entry.config(state=tk.DISABLED) # Disable the user input
        self.send_button.config(state=tk.DISABLED) # Disable the send button
        # Get the initial system context
        initial_greeting_context, seasonal_offer_indices = self.agent.build_rag_initial_context(k=self.initial_greeting_k)
        for article_id in seasonal_offer_indices:
            print(self.agent.preprocessed_hm_articles.loc[article_id, "combined"])
        # Add the initial fake message to give the context
        self.agent.add_message(role="system", content=initial_greeting_context)
        # Get the response
        initial_response = self.agent.generate_rag_response(doc_indices=seasonal_offer_indices)
        # Add the response to the message history
        self.agent.add_message(role="assistant", content=initial_response)
        # Get the image paths
        image_paths = []
        for article_id in seasonal_offer_indices:
            image_paths.append(self.agent.get_image_path(article_id))
        # Display the response and the images
        self.root.after(0, lambda: self.update_dialog_display_with_images(initial_response, image_paths)) 

    def handle_image_upload(self):
        """
        This function handles the file upload process.
        """
        pass

    def handle_user_input(self, event=None):
        """
        This function handles the user input and update the conversation display.
        """
        user_input = self.text_entry.get("1.0", tk.END).strip() # Remove the spaces
        if user_input:
            # Display the user input
            self.dialog_display.config(state=tk.NORMAL)
            self.dialog_display.insert(tk.END, f"You: {user_input}\n")
            self.dialog_display.config(state=tk.DISABLED)
            self.dialog_display.yview(tk.END)  # Scroll to the end

            # Empty the user input
            self.text_entry.delete("1.0", tk.END)
            self.text_entry.config(state=tk.DISABLED) # Disable the user input
            self.send_button.config(state=tk.DISABLED) # Disable the send button

            # Get the response from the AI agent in an individual thread (To not block the text display)
            threading.Thread(target=self.process_ai_response, args=(user_input,)).start()

    def process_ai_response(self, user_input):
        """
        This function process the user input and display the AI response.

        user_input: The input text from the user.
        """
        # Determine the question type
        question_type = self.agent.analyze_question_type(user_input)
        print(question_type)
        # Determine if the RAG should be applied and generate context-included input
        modified_user_input = None
        if question_type == "other":
            # Set the user input
            modified_user_input = user_input
            # Add the message to the agent history
            self.agent.add_message(role="user", content=modified_user_input)
            # Get the response
            response = self.agent.generate_response()
            # Add the response to the message history
            self.agent.add_message(role="assistant", content=response)
            # Display the response
            self.root.after(0, self.update_dialog_display, response)
        else:
            # Set the user input
            modified_user_input, article_indices = self.agent.build_rag_user_input_keyword_match(user_input, k=self.rag_k_keyword)
            for article_id in article_indices:
                print(self.agent.preprocessed_hm_articles.loc[article_id, "combined"])
            # Add the message to the agent history
            self.agent.add_message(role="user", content=modified_user_input)
            # Get the response
            response = self.agent.generate_rag_response(doc_indices=article_indices)
            # Add the response to the message history
            self.agent.add_message(role="assistant", content=response)
            # Get the image paths
            image_paths = []
            for article_id in article_indices:
                image_paths.append(self.agent.get_image_path(article_id))
            # Display the response and the images
            self.root.after(0, lambda: self.update_dialog_display_with_images(response, image_paths)) 

    def update_dialog_display(self, response):
        """
        This function update the dialog according to the response gathered from the AI model.
        """
        # Update the response
        self.dialog_display.config(state=tk.NORMAL)
        self.dialog_display.insert(tk.END, f"Assistant: {response}\n")
        self.dialog_display.config(state=tk.DISABLED)
        self.dialog_display.yview(tk.END)  # Scroll to the end
        # Unlock the user input and the send button
        self.text_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)

    def update_dialog_display_with_images(self, response, image_paths):
        """
        This function update the dialog according to the response gathered from the AI model.
        """
        # Update the response
        self.dialog_display.config(state=tk.NORMAL)
        self.dialog_display.insert(tk.END, f"Assistant: {response}\n")
        for image_path in image_paths:
            self.insert_image(image_path)
        self.dialog_display.config(state=tk.DISABLED)
        self.dialog_display.yview(tk.END)  # Scroll to the end
        # Unlock the user input and the send button
        self.text_entry.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)

    def insert_image(self, image_path):
        """
        This function insert an image into the dialog.

        image_path: The path to the image to be inserted.
        """
        image_existence = content_existence(image_path)
        if image_existence:
            # Open and resize the image
            image = Image.open(image_path)
            image_copy = image.copy()  # Create a copy of the image
            image_copy.thumbnail(max_image_size) 
            photo = ImageTk.PhotoImage(image_copy)

            # Insert the image into the text entry
            self.dialog_display.image_create(tk.END, image=photo)
            self.dialog_display.insert(tk.END, "\n")  # Add a newline after the image

            # Add the image to the references
            self.image_references.append(photo)