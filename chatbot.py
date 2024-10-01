import json
import os
import shutil
from uuid import uuid1
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import glob
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage,AIMessage
from datetime import datetime
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import constants
from constants import llm_s,llm_r
from openai import OpenAI

client = OpenAI()

def format_conversation(messages):
    conversation_str = ""

    for message in messages:
        if isinstance(message, HumanMessage):
            conversation_str += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_str += f"AI: {message.content}\n"

    return conversation_str.strip()


def convert_to_chat_list(input_string:str):
    chat_list = []
    lines = input_string.splitlines()

    for line in lines:
        if line.startswith("Human:"):
            chat_list.append({"role": "user", "content": line.replace("Human: ", "").strip()})
        elif line.startswith("AI:"):
            chat_list.append({"role": "assistant", "content": line.replace("AI: ", "").strip()})

    return chat_list

def get_gpt_response_query(
    memory_messages,
    prompt:str,
    query: str,
    model: str ,
    temprature: float = 0,
    max_tokens: int = 4000,
):
    prompt_set = prompt.split('Below is the chat history:')
    messages = [{"role":"system","content": prompt_set[0]}] + convert_to_chat_list(memory_messages) + [{"role": "user", "content": query}]

    try:
        response= client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temprature,
                max_tokens=max_tokens,
            )
        output = response.choices[0].message.content
        return output
    except Exception as e:
        return str(e)


class Chatbot:
    def __init__(
            self,
            documents,
            text_splitter: str = constants.DEFAULT_TEXT_SPLITTER,
            chunk_size: int = constants.CHUNK_SIZE,
            chunk_overlap: int = constants.CHUNK_OVERLAP,
            embedding_model=constants.DEFAULT_EMBEDDING_MODEL,
            vector_store=constants.DEFAULT_VECTOR_STORE,
            prompt_for_rephrase_query=constants.PROMPT_FOR_REPHRASE_QUERY,
            prompt_for_rephrase_query_variables=constants.PROMPT_FOR_REPHRASE_QUERY_VARIABLES,
            prompt_for_summary = constants.PROMPT_FOR_SUMMARY,
            prompt_for_summary_variables = constants.PROMPT_FOR_SUMMARY_VARIABLES,
            retriever_method=constants.DEFAULT_RETRIEVAL_METHOD,
            retriever_search_type=constants.DEFAULT_RETRIEVAL_SEARCH_TYPE,
            retriever_top_k=constants.DEFAULT_RETRIEVER_TOP_K,
            retriever_similarity_score_threshold=constants.DEFAULT_RETRIEVER_SIMILARITY_SCORE_THRESHOLD,
            prompt_for_generation=constants.PROMPT_FOR_GENERATION,
            prompt_for_generation_variables=constants.PROMPT_FOR_GENERATION_VARIABLES,
            prompt_to_be_added = "",
            memory_type=constants.DEFAULT_MEMORY_TYPE,
            memory_last_k=constants.MEMORY_LAST_K,
            memory_max_tokens=constants.MEMORY_MAX_TOKENS,
            llm_model=constants.DEFAULT_LLM,
            verbose=False,
            load_chatbot_from_dir=None,
            memory_filename = f'memory_at_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.json'
        ):
        self.documents = documents
        self.text_splitter_name = text_splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        self.vector_store_name = vector_store
        self.prompt_for_rephrase_query = prompt_for_rephrase_query
        self.prompt_for_rephrase_query_variables = prompt_for_rephrase_query_variables
        self.prompt_for_summary = prompt_for_summary
        self.prompt_for_summary_variables = prompt_for_summary_variables
        self.retriever_search_type = retriever_search_type
        self.retriever_top_k=retriever_top_k
        self.retriever_similarity_score_threshold = retriever_similarity_score_threshold
        self.prompt_for_generation = prompt_for_generation
        self.prompt_to_be_added = prompt_to_be_added
        self.prompt_for_generation_variables = prompt_for_generation_variables if self.documents != [] else ['chat_history','input'] 
        self.memory_type = memory_type
        self.memory_last_k = memory_last_k
        self.memory_max_tokens = memory_max_tokens
        self.llm_model_name = llm_model
        self.verbose = verbose
        self.load_chatbot_from_dir = load_chatbot_from_dir 
        self.retriever_method = retriever_method           
        self.summary_prompt = PromptTemplate(template=self.prompt_for_summary, input_variables=['response'])
        self.summarize_chain = LLMChain(llm=llm_s,prompt=self.summary_prompt)
        self.prompt_template = PromptTemplate(template=self.prompt_for_generation + self.prompt_to_be_added,
                                              input_variables=self.prompt_for_generation_variables)
        self.llm = self._get_llm()
        self.vector_store = self._init_vector_store()
        self.retriever = self._get_retriever(self.vector_store, self.retriever_search_type,
                                             self.retriever_top_k, self.retriever_similarity_score_threshold,
                                             self.retriever_method, self.llm)
        self.memory = self._get_memory()
        self.memory_filename = memory_filename
        self.qa_chain = load_qa_chain(self.llm,
                                      chain_type='stuff',
                                      prompt=self.prompt_template,
                                      memory=self.memory) if self.documents else LLMChain(llm=self.llm, prompt=self.prompt_template,memory=self.memory ,verbose=self.verbose)

    
    def query(self, query: str):
        if self.memory.chat_memory != '' and self.documents:
            query = self._rephrase_query(query)
            
        if self.documents:
            docs, scores = self._get_docs(query)
        else:
            docs,scores = [],[]
            
        
        with get_openai_callback() as cb:
            if self.documents:
                res = self.qa_chain.invoke({self.prompt_for_generation_variables[-1]: query,'input_documents':docs}, return_only_outputs=True)
                output_full = res.get('output_text').strip()
            else:
                messages_list = self.memory.chat_memory.messages
                memory_messages = format_conversation(messages_list)
                output_full = get_gpt_response_query(prompt=self.prompt_for_generation,model=self.llm_model_name,query=query,memory_messages=memory_messages)
                self.memory.save_context({"input": query}, {"output": output_full})
        # output = self.summarize_chain.invoke(output_full)['text']
        return output_full, docs, scores
    

    def clear_memory(self):
        self.memory.clear()


    def _get_docs(self, query):
        docs = self.retriever.get_relevant_documents(query)

        query_embed = self.embeddings_model.embed_query(query)
        docs_embed = self.embeddings_model.embed_documents([doc.page_content for doc in docs])

        if len(docs) == 0:
            return [], []

        query_embed = np.array(query_embed).reshape(1, -1)
        docs_embed = np.array(docs_embed)
        scores = cosine_similarity(query_embed, docs_embed)[0]

        return docs, scores


    def _rephrase_query(self, query):
        prompt_temp = PromptTemplate(template=self.prompt_for_rephrase_query,
                                     input_variables=self.prompt_for_rephrase_query_variables)
        prompt_chain = LLMChain(llm=llm_r, prompt=prompt_temp, verbose=self.verbose)
        memory  = self.memory.load_memory_variables({})['chat_history']
        rephrased_query = prompt_chain.invoke({'memory': memory, 'query': query})
        print(rephrased_query['text'])
        return rephrased_query['text']


    def _read_pdf_and_make_chunks(self,docs_list):
        if 'Recursive' in self.text_splitter_name:
            self.text_splitter = RecursiveCharacterTextSplitter(separators=[' ','\n'],
                                                                     chunk_size=self.chunk_size,
                                                                     chunk_overlap=self.chunk_overlap)
        elif 'Token' in self.text_splitter_name:
            self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size,
                                                        chunk_overlap=self.chunk_overlap)
        elif 'Semantic' in self.text_splitter_name:
            self.text_splitter = SemanticChunker(
            self.embeddings_model, breakpoint_threshold_type="percentile")

        else:
            self.text_splitter = CharacterTextSplitter(separator=' ',
                                                            chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)

        docs = []
        for doc_path in docs_list:
            raw_context = ""
            if doc_path.endswith('txt') or doc_path.endswith('pdf'):
                print('txt/pdf',doc_path)
                loader = PyMuPDFLoader(doc_path)
            
            elif doc_path.endswith('docx') or doc_path.endswith('doc'):
                print('docx',doc_path)
                loader = Docx2txtLoader(doc_path)
            
            documents = loader.load()    
            for page in documents:
                raw_context+=page.page_content+'\n'

            book = [Document(page_content=raw_context,metadata={'name':doc_path})]
            chunks = self.text_splitter.split_documents(book)
            chunk_no = 1
            for chunk in chunks:
                chunk.metadata['Section'] = chunk_no
                docs.append(chunk)
                chunk_no +=1

        return docs


    def _init_vector_store(self):
        if 'Google' in self.embedding_model_name:
            self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        elif 'text-embedding' in self.embedding_model_name:
            self.embeddings_model = OpenAIEmbeddings(model=self.embedding_model_name)

        if self.load_chatbot_from_dir:
            self.vectorstore_dir = self.load_chatbot_from_dir+'/vectorstore'
            if os.path.exists(self.vectorstore_dir):
                if self.vector_store_name == 'faiss':
                    vector_store = FAISS.load_local(self.vectorstore_dir, self.embeddings_model)
                elif self.vector_store_name == 'chroma':
                    vector_store = Chroma(persist_directory=self.vectorstore_dir,
                                        embedding_function=self.embeddings_model)
            else:
                vector_store = None
            return vector_store

        if not self.documents:
            # Create an empty vector store if no documents are provided
            if self.vector_store_name == 'faiss':
                vector_store = FAISS.from_texts([''],self.embeddings_model)
            elif self.vector_store_name == 'chroma':
                vector_store = Chroma(embedding_function=self.embeddings_model)
            return vector_store

        chunks = self._read_pdf_and_make_chunks(self.documents)

        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        self.vectorstore_dir = './tmp/'+str(uuid1())

        if self.vector_store_name == 'faiss':
            vector_store = FAISS.from_documents(chunks, embedding=self.embeddings_model)
            vector_store.save_local(self.vectorstore_dir)
        elif self.vector_store_name == 'chroma':
            vector_store = Chroma.from_documents(chunks, embedding=self.embeddings_model, persist_directory=self.vectorstore_dir)
        
        return vector_store

    def delete_doc_from_vstore(self,name_of_deletion):
        def faiss_to_df(store): # convert vectore store to dataframe to easily get target ids
                store_dict = store.docstore._dict
                dict_rows = []
                for key in store_dict.keys():
                    name = store_dict[key].metadata['name']
                    dict_rows.append({'id':key,'name':name})
            
                return pd.DataFrame(dict_rows) # df with ids and paths
            
        def delete_doc_from_faiss(store,name_of_deletion): # deletes in place
            vector_df = faiss_to_df(store)
            ids_to_delete = vector_df.loc[vector_df['name']==name_of_deletion]['id'].tolist() # list of ids to delete, only where id matches to deletion path
            store.delete(ids_to_delete)
            return ids_to_delete

        def delete_doc_from_chroma(chroma_store,name_of_deletion):
            def extract_name(metadata):
                return metadata.get('name')
        
            vector_df = pd.DataFrame(chroma_store.get())
            vector_df['name'] = vector_df['metadatas'].apply(lambda x: extract_name(x))
            ids_to_delete = vector_df.loc[vector_df['name']==name_of_deletion]['ids'].tolist() # list of ids to delete, only where id matches to deletion path
            chroma_store._collection.delete(ids = ids_to_delete) 
            return ids_to_delete

        if self.vector_store_name == 'faiss':
            delete_doc_from_faiss(self.vector_store,name_of_deletion)
            self.vector_store.save_local(self.vectorstore_dir)
        elif self.vector_store_name == 'chroma':
            delete_doc_from_chroma(self.vector_store,name_of_deletion)
            self.vector_store.persist()
        print('Deleted Docs')

    def add_docs_to_vstore(self):
            
        new_docs_dir = os.path.join(self.load_chatbot_from_dir, 'docs', 'tmp')
        new_pdfs = [os.path.join(new_docs_dir, f) for f in os.listdir(new_docs_dir) if f.endswith('.txt') or f.endswith('.pdf') or f.endswith('.docx')]
        # print(new_pdfs)
        old_docs_dir = os.path.join(self.load_chatbot_from_dir, 'docs')
        old_pdfs = [os.path.join(old_docs_dir, f) for f in os.listdir(old_docs_dir) if f.endswith('.txt') or f.endswith('.pdf') or f.endswith('.docx')]
        # print(old_pdfs)
        new_pdfs = [f for f in new_pdfs if os.path.basename(f) not in [os.path.basename(x) for x in old_pdfs]]
        if new_pdfs != []:
            new_docs = self._read_pdf_and_make_chunks(new_pdfs)
    
            
            if self.vector_store_name == 'faiss':
                dc = FAISS.from_documents(new_docs,self.embeddings_model)
                self.vector_store.merge_from(dc)
                self.vector_store.save_local(self.vectorstore_dir)
            
        
            elif self.vector_store_name == 'chroma':
                self.vector_store.add_documents(new_docs)
                self.vector_store.persist()
    
            for doc_path in new_pdfs:
                # print(doc_path)
                print(self.load_chatbot_from_dir + '/docs/')
                try:
                    shutil.move(doc_path,self.load_chatbot_from_dir + '/docs/')
                    os.remove(doc_path)
                except:
                    pass
                
        print('Added Docs')
        
    
    def _get_retriever(self, vector_store, retriever_search_type, top_k, similarity_score_threshold, retriever_method, llm_model):
        if not vector_store:
            return None

        retriever_config = {
            "Similarity score threshold": ("similarity_score_threshold", {"score_threshold": similarity_score_threshold}),
            "top_k": ("similarity", {"k": top_k}),
            "mmr": ("mmr", {})
        }

        search_type, search_kwargs = retriever_config.get(retriever_search_type, (None, {}))
        if search_type:
            retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        else:
            raise ValueError(f"Invalid retriever_search_type: {retriever_search_type}")

        if retriever_method == "MultiQueryRetriever":
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm_model)
        elif retriever_method == "Contextual compression":
            compressor = LLMChainExtractor.from_llm(llm_model)
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        elif retriever_method != "Default":
            raise ValueError(f"Invalid retriever_method: {retriever_method}")

        return retriever



    def _get_llm(self):
        if 'Google' in self.llm_model_name:
            llm = GoogleGenerativeAI(model="gemini-pro")
        else:
            llm = ChatOpenAI(temperature=0,
                             max_tokens=4000,
                             model=self.llm_model_name,
                             request_timeout=120)
        return llm


    def _get_memory(self):
        if self.memory_type == 'ConversationBufferMemory':
            memory = ConversationBufferMemory(memory_key='chat_history', input_key=self.prompt_for_generation_variables[-1])
        elif self.memory_type == 'ConversationBufferWindowMemory':
            memory = ConversationBufferWindowMemory(memory_key='chat_history', input_key=self.prompt_for_generation_variables[-1], k=self.memory_last_k)
        elif self.memory_type == 'ConversationTokenBufferMemory':
            memory = ConversationTokenBufferMemory(llm=self.llm, memory_key='chat_history', input_key=self.prompt_for_generation_variables[-1], max_token_limit=self.memory_max_tokens)
        elif self.memory_type == 'ConversationSummaryMemory':
            memory = ConversationSummaryMemory(llm=self.llm, memory_key='chat_history', input_key=self.prompt_for_generation_variables[-1])

        return memory


    def clear_memory(self):
        self.memory.clear()


    def save_docs_embeddings(self, dir_path):
        shutil.copytree(self.vectorstore_dir, dir_path)


    @staticmethod
    def load_chatbot(dir_path):
        with open(dir_path+'/settings.json') as f:
            settings = json.load(f)

        print('Chatbot Initialized')
        chatbot = Chatbot(**settings, load_chatbot_from_dir=dir_path)
        return chatbot


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    chatbot = Chatbot(
        documents=['docs/example.pdf'],
        verbose=False
    )

    chatbot.query('what is the total number of AI publications?')
    chatbot.query('What is this number divided by 2?')

