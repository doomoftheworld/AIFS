# AI Fashion Stylist (AIFS)
## This repo contains:
- The code for launching the conversation with the AI agent.
- A few test images are in the folder "upload_test_images" to test the functionality of searching products based on user-provided images.

## Instruction for executing AIFS:
### Prerequisite:
To execute the AIFS CHAT application, you must have the following software installed (the versions are the recommended ones):

- Python 3.12.4
- CUDA with cudnn (To support the corresponding pytorch platform), you should have an NVIDIA graphics card (e.g., CUDA 11.8 with NVIDIA GeForce RTX 4080 Laptop GPU).
- Pytorch platform that is compatible with the CUDA installed.

The required libraries are listed in the file "requirements.text".

### The precomputed embeddings and the H&M dataset.
It is important to download the H&M Fashion dataset and the precomputed embeddings for the RAG search:
- Download the precomputed embeddings from https://zenodo.org/records/15090151 and extract it to the folder "precomputed_data" (It should be in the path AIFS\\precomputed_data).
- Download the H&M Personalized Fashion Recommendations dataset from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations and extract it to the folder "h-and-m-personalized-fashion-recommendations" (It should be in the path AIFS\\h-and-m-personalized-fashion-recommendations).

### Place the OpenAI API key
It is also important to configure a system environment variable named "OPENAI_API_KEY" that contains your OpenAI API key.

### Launch the chat application
Open a terminal and execute "python main.py".

### Features of the application
- The agent will recommend fashion items based on the H&M inventory it has.
- You can use the bottom text entry to give the agent your desired message.
- You can use the "Enter" key or click on the button "Send" to send the message.
- The "Upload Image" button is for cases where you would like to provide an example of the fashion item you want to find.

### Example of conversation
![alt text](https://github.com/doomoftheworld/AIFS/chat_examples/Frame1.png "Conversation turn 1")
![alt text](https://github.com/doomoftheworld/AIFS/chat_examples/Frame2.png "Conversation turn 2")
![alt text](https://github.com/doomoftheworld/AIFS/chat_examples/Frame3.png "Conversation turn 3")
![alt text](https://github.com/doomoftheworld/AIFS/chat_examples/Frame4.png "Conversation turn 4-1")
![alt text](https://github.com/doomoftheworld/AIFS/chat_examples/Frame5.png "Conversation turn 4-2")
