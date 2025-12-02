"""
Baseline RAG System for Graph-Augmented RAG Research Project
Week 9: Baseline Implementation
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Use OpenAI directly
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Legacy imports (commented out for reference)
#from openrouter import OpenRouterChat
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_classic.chat_models import ChatOpenAI



# LlamaIndex imports (optional alternative)
from llama_index.core import VectorStoreIndex, Document as LlamaDocument
from llama_index.core.retrievers import VectorIndexRetriever


@dataclass
class RAGResponse:
    """Structure for RAG system responses"""
    query: str
    answer: str
    retrieved_docs: List[str]
    metadata: Dict[str, Any]


class BaselineRAG:
    """
    Baseline Retrieval-Augmented Generation System
    
    This serves as the baseline for comparison against Graph-Augmented RAG.
    Uses standard retrieval → generation pipeline without graph reasoning.
    """
    
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",  # Changed to OpenAI model
        # api_base: str = "https://openrouter.ai/api/v1",  # REMOVED: Not needed for OpenAI
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k_retrieve: int = 4
    ):
        """
        Baseline RAG with OpenAI API

        Args:
            api_key: OpenAI API key
            model_name: OpenAI model (e.g., gpt-4o-mini, gpt-4, gpt-3.5-turbo)
            chunk_size: Token chunk size for splitting
            chunk_overlap: Overlap between chunks
            k_retrieve: Number of documents to retrieve
        """
        self.api_key = api_key
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieve = k_retrieve

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key

        # Use OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()

        # Chat LLM using OpenAI API
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        
        # ===== COMMENTED OUT: Old OpenRouter implementation =====
        # # Pass environment variables for OpenRouter-compatible API
        # os.environ["OPENAI_API_BASE"] = api_base
        # # Use HuggingFace embeddings (no OpenAI calls)
        # self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # # Chat LLM using OpenRouter API
        # self.llm = ChatOpenAI(
        #     model_name=model_name,
        #     temperature=0,
        #     openai_api_key=api_key,
        #     openai_api_base=api_base  # points to OpenRouter
        # )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.vectorstore = None
        self.qa_chain = None
    
    # def __init__(
    #     self,
    #     api_key: str,
    #     model_name: str = "deepseek/deepseek-chat",
    #     api_base: str = "https://openrouter.ai/api/v1",
    #     chunk_size: int = 500,
    #     chunk_overlap: int = 50,
    #     k_retrieve: int = 4
    # ):
    #     self.api_key = api_key
    #     self.model_name = model_name
    #     self.chunk_size = chunk_size
    #     self.chunk_overlap = chunk_overlap
    #     self.k_retrieve = k_retrieve

    #     os.environ["OPENAI_API_KEY"] = api_key  # still required by LangChain
    #     os.environ["OPENAI_API_BASE"] = api_base

    #     # Use embeddings & chat LLM from OpenRouter via OpenAI-compatible interface
    #     self.embeddings = OpenAIEmbeddings()  
    #     self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        
    #     self.text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap
    #     )

    #     self.vectorstore = None
    #     self.qa_chain = None
    
    # def __init__(
    #     self,
    #     openai_api_key: str,
    #     model_name: str = "gpt-3.5-turbo",
    #     chunk_size: int = 500,
    #     chunk_overlap: int = 50,
    #     k_retrieve: int = 4
    # ):
    #     """
    #     Initialize Baseline RAG System
        
    #     Args:
    #         openai_api_key: OpenAI API key
    #         model_name: LLM model to use
    #         chunk_size: Size of text chunks for splitting
    #         chunk_overlap: Overlap between chunks
    #         k_retrieve: Number of documents to retrieve
    #     """
    #     self.openai_api_key = openai_api_key
    #     os.environ["OPENAI_API_KEY"] = openai_api_key
        
    #     self.model_name = model_name
    #     self.chunk_size = chunk_size
    #     self.chunk_overlap = chunk_overlap
    #     self.k_retrieve = k_retrieve
        
    #     # Initialize components
    #     self.embeddings = OpenAIEmbeddings()
    #     self.llm = ChatOpenAI(model_name=model_name, temperature=0)
    #     self.text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap
    #     )
        
    #     self.vectorstore = None
    #     self.qa_chain = None
        
    def load_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Load and process documents into vector store
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        # Create Document objects
        docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas or [{}] * len(documents))
        ]
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(docs)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        
        print(f"✓ Loaded {len(documents)} documents")
        print(f"✓ Created {len(split_docs)} chunks")
        print(f"✓ Vector store initialized")
        
    def build_qa_chain(self):
        """Build the QA chain with custom prompt"""
        
        # Custom prompt template for better control
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: Let me provide a factual answer based on the context."""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.k_retrieve}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("✓ QA chain built successfully")
        
    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG system
        
        Args:
            question: User query
            
        Returns:
            RAGResponse with answer and metadata
        """
        if not self.qa_chain:
            raise ValueError("QA chain not built. Call build_qa_chain() first.")
        
        # Get response
        result = self.qa_chain.invoke({"query": question})
        
        # Extract retrieved document contents
        retrieved_docs = [
            doc.page_content for doc in result.get("source_documents", [])
        ]
        
        # Create structured response
        response = RAGResponse(
            query=question,
            answer=result["result"],
            retrieved_docs=retrieved_docs,
            metadata={
                "num_retrieved": len(retrieved_docs),
                "model": self.model_name
            }
        )
        
        return response

    def retrieve_similar_documents(self, query: str, k: Optional[int] = None) -> List[str]:
        """Expose vector similarity search for downstream graph augmentation."""
        if not self.vectorstore:
            return []
        k = k or self.k_retrieve
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def evaluate_hallucination(self, response: RAGResponse) -> Dict[str, Any]:
        """
        Basic hallucination detection
        
        Checks if answer contains information not in retrieved documents
        (Simplified version - can be enhanced with NLI models)
        """
        answer_lower = response.answer.lower()
        
        # Check for common hallucination phrases
        uncertain_phrases = [
            "i don't know",
            "not mentioned",
            "no information",
            "cannot determine"
        ]
        
        is_uncertain = any(phrase in answer_lower for phrase in uncertain_phrases)
        
        # Calculate overlap between answer and retrieved docs
        retrieved_text = " ".join(response.retrieved_docs).lower()
        
        # Simple word overlap metric
        answer_words = set(answer_lower.split())
        retrieved_words = set(retrieved_text.split())
        
        overlap_ratio = len(answer_words & retrieved_words) / len(answer_words) if answer_words else 0
        
        return {
            "is_uncertain": is_uncertain,
            "word_overlap_ratio": overlap_ratio,
            "likely_hallucination": overlap_ratio < 0.3 and not is_uncertain
        }


# Example usage and testing
if __name__ == "__main__":
    # Sample documents for testing
    sample_docs = [
        "Albert Einstein was a theoretical physicist born in 1879 in Germany. He developed the theory of relativity.",
        "The theory of relativity consists of special relativity and general relativity. Special relativity was published in 1905.",
        "Einstein received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
        "Einstein moved to the United States in 1933 and became a professor at Princeton University."
    ]
    
    # Initialize RAG system
    # NOTE: Replace with your actual API key
    API_KEY = "your-openai-api-key-here"
    
    rag = BaselineRAG(
        api_key=API_KEY,
        model_name="gpt-4o-mini",
        k_retrieve=3
    )
    
    # Load documents
    rag.load_documents(sample_docs)
    
    # Build QA chain
    rag.build_qa_chain()
    
    # Test queries
    test_queries = [
        "When was Einstein born?",
        "What prize did Einstein receive?",
        "What is the theory of relativity?"
    ]
    
    print("\n" + "="*60)
    print("BASELINE RAG SYSTEM - TEST RESULTS")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        response = rag.query(query)
        print(f"🤖 Answer: {response.answer}")
        
        # Evaluate for hallucination
        eval_result = rag.evaluate_hallucination(response)
        print(f"📊 Evaluation: {json.dumps(eval_result, indent=2)}")
        print("-" * 60)