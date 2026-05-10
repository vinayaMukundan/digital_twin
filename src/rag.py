import os
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate



class VitaTwinRAG:
    def __init__(self):
        print("--- Initializing RAG Pipeline ---")
        
        # 1. Setup Local Embeddings (free, runs locally)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Free HuggingFace LLM
        hf_token = os.getenv("HF_TOKEN")
        print(f"--- HF Token present: {bool(hf_token)} ---")
        if not hf_token:
            raise ValueError("HF_TOKEN is MISSING from Space secrets")
        
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                huggingfacehub_api_token=hf_token,
                max_new_tokens=512,
                temperature=0.3,
                task="conversational",
                provider="groq",
            )
        )
        
        # 3. Prompt Template
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a clinical assistant. Use the following context from a patient note to answer the question. If you don't know the answer based on the note, say that you don't know. Context: {context}"),
            ("human", "{question}")
        ])
        
    def query(self, question: str, patient_note: str):
        # A. Chunk the note
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(patient_note)
        
        # B. Create temporary vector store
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            collection_name="temp_patient_note"
        )
        
        # C. Build RAG chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # D. Get answer and cleanup
        try:
            response = rag_chain.invoke(question)
        finally:
            vectorstore.delete_collection()
        
        return {"answer": response}
