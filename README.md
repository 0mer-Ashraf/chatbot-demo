## Demo UI

<img width="1379" alt="Screenshot 2024-08-28 at 4 36 07 PM" src="https://github.com/user-attachments/assets/ebed81bf-e097-4754-82ba-d92732ffb8da">


## Introduction

Bot-in-a-Box is a configurable chabot, with a vast amount of options and each one is customizable. The main functionality is for the user to be able to upload document(s) of their choice and then be able to chat with them. It is a RAG pipeline which consists of three main components:
1. Indexing of Uploaded Documents
2. Retrieval of Relevant Docs
3. Synthesis of Response

The application is built using ```Langchain``` and ```Streamlit```.

## Breaking Down the Components

**1. Document Upload**

The allowed file types in the finished version of the app will be: pdf, txt, word, html, json, markdown. For now only PDFs are allowed.

<img width="303" alt="Screenshot 2024-08-28 at 4 38 33 PM" src="https://github.com/user-attachments/assets/158939d8-a7d5-4d23-a622-9e0b45f9a1e5">


**2. Splitter Selection**

Use LangChain to break documents into manageable pieces, as LLMs have input size limitations. Splitting consists of the type of splitter being used and splitting hyperparameters.

- CharacterTextSplit: This splits based on Separator(by default “”) and measures chunk length by number of characters.
- RecursiveTextSplit: The recursive text splitter works by recursively splitting text into smaller chunks based on a list of separators
- TokenTextSplit: Splitting is done by a number of tokens for which any tokenizer can be used. E.g NLTK, Spacy, and tiktoken.
- Semantic Chunking: Splits the text based on semantic similarity; this splits into sentences, then groups into groups of 3 sentences, and then merges one that is similar in the embedding space.

<img width="261" alt="Screenshot 2024-08-28 at 4 38 57 PM" src="https://github.com/user-attachments/assets/687905f6-c906-4256-88e6-a18960c66166">


**3. Embedding Model Selection**

Convert the chunks into numerical representations via embedding models and store them in a vector database for retrieval. The chunk size and overlap options are configurable as well. The embedding models will be configurable from a wide range of options, inclusing OpenAI Models, Google Palm Model as well as top models from Huggingface MTEB Leaderboard. e.g:

- text-embedding-3-small
- text-embedding-3-large
- text-embedding-ada-002
- bge-base-1.5
- bge-large-1.5

Few models require deploying on own server, so they are not available in the current version of the app.

<img width="290" alt="Screenshot 2024-08-28 at 4 39 17 PM" src="https://github.com/user-attachments/assets/98075123-6649-40db-93f7-390548af2c18">


**4. Vector Store Selection**

After embedding the chunks, these vector need to be stored in a Vector Database, for efficient storing and retrieval. The right vectorstore depend on the use case and amount of data. The user can select the vector store from the following:

- FAISS
- ChromaDB
- Pinecone
- Qdrant

Pinecone and Qdrant require self hosting, so for now only FAISS and ChromaDB can be selected.

<img width="272" alt="Screenshot 2024-08-28 at 4 43 25 PM" src="https://github.com/user-attachments/assets/8ec98d5b-65ea-42f5-8111-8ed80e63b979">


**5. Retrieval**

The retriever retrieves document chunks that best match the user's query. There are mainly two types of retrieval: Distance-based retrieval and LLM-aided retrieval. Distance based retrieval then has several choices, including:

- Top k (retrieving the top k number of docs, no matter how similar or disimilar)
- MMR (The MMR selects examples based on a combination of which examples are most similar to the inputs, while also optimizing for diversity)
- Similarity Threshold (setting a threshold for cosine similarity and then retrieving all docs above that threshold)

<img width="257" alt="Screenshot 2024-08-28 at 4 43 43 PM" src="https://github.com/user-attachments/assets/2c948ea0-263d-476c-a4f7-aa506dd8920b">

LLM aided retrieval has a few option like:
- Compression
- Map reduce
- Refine
- Map rerank

**6. Generation**

The generation model are the models which synthesize response, from the question and the context. The choice of this model usually depends on the speed-accuracy tradeoff which is acceptable to the user. Another factor is self-hosted model vs API-based model. A final factor can also be context window of the model. Based on these factors, the options for LLMs can be:

- GPT 3.5 Turbo
- GPT 3.5 Turbo 16k
- GPT 4
- GPT 4 Turbo 128k
- Google Palm
- Claude
- LLaMa 2 (7B, 13B)
- Mistral 7B

For now only the top API based models are available, i.e GPT 3.5 Turbo 16k, GPT 4 Turbo 128k, GPT-4o.

<img width="268" alt="Screenshot 2024-08-28 at 4 44 23 PM" src="https://github.com/user-attachments/assets/7f8fcbce-434f-49b8-8a43-c5adb1bff302">

## Future Improvements

The future development mainly consists of adding the remaining selectable options for more customization. The option for configuring the memory is also under consideration, i.e choice between ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory etc.


