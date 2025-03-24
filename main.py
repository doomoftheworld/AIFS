import tkinter as tk
from common_imports import os
from AIFS_window import AIFSChatApp
from AIFS_agent import AIFSAgent

"""
The main function
"""
if __name__ == "__main__":
    # Create the window
    root = tk.Tk()
    # Start the conversaton
    chat_app = AIFSChatApp(root=root, api_key=os.getenv("OPENAI_API_KEY"))
    # chat_agent = AIFSAgent(api_key=os.getenv("OPENAI_API_KEY"), chat_model="gpt-3.5-turbo", embedding_model="text-embedding-3-small")
    root.mainloop()