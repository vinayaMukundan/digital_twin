from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
class VitaTwinRAG:
    def __init__(self):
        print("--- Initializing RAG Pipeline ---")
        # 1. Setup Local Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Setup LLM
        self.llm = Ollama(model="llama3")
        # 3. Define the Prompt Template
        template = """
        You are a clinical assistant. Use the following pieces of context from a patient note to answer the question. 
        If you don't know the answer based on the note, say that you don't know.
        
        Context: {context}
        Question: {question}
        
        Helpful Clinical Answer:"""
        self.prompt = PromptTemplate.from_template(template)

    def query(self, question: str, patient_note: str):
        # A. Process the specific note (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(patient_note)

        # B. Create a volatile Vector DB
        vectorstore = Chroma.from_texts(
            texts=texts, 
            embedding=self.embeddings,
            collection_name="temp_patient_note"
        )

        # C. Build Modern Retrieval Chain (LCEL)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # This replaces RetrievalQA
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # D. Get Answer
        try:
            response = rag_chain.invoke(question)
        finally:
            # E. Cleanup
            vectorstore.delete_collection()
        
        return {"answer": response}