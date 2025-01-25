# app.py
import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import pipeline

# Initialize models and Pinecone once
@st.cache_resource
def init_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("question-answering", model="google/flan-t5-large")
    return embedding_model, qa_model

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc

def process_pdfs(pdfs, embedding_model, pc):
    # Create or connect to index
    index_name = st.secrets["PINECONE_INDEX"]
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
    
    index = pc.Index(index_name)
    
    # Process each PDF
    for pdf_file in pdfs:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.getbuffer())
            loader = PyPDFLoader(tmp.name)
            pages = loader.load()
            os.unlink(tmp.name)
        
        # Split text
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separator="\n"
        )
        
        chunks = []
        for page_num, page in enumerate(pages):
            page_chunks = text_splitter.split_text(page.page_content)
            for chunk_num, chunk in enumerate(page_chunks):
                chunks.append({
                    "text": chunk,
                    "source": pdf_file.name,
                    "page": page_num + 1,
                    "chunk_num": chunk_num
                })
        
        # Generate embeddings
        embeddings = embedding_model.encode([chunk["text"] for chunk in chunks])
        
        # Prepare vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": f"{chunk['source']}_p{chunk['page']}_c{chunk['chunk_num']}",
                "values": embeddings[i].tolist(),
                "metadata": chunk
            })
        
        # Upload to Pinecone
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i+100])

def main():
    st.title("PDF Knowledge Assistant 📚")
    st.markdown("Upload PDFs and ask questions about their content")
    
    # Initialize components
    embedding_model, qa_model = init_models()
    pc = init_pinecone()
    index = pc.Index(st.secrets["PINECONE_INDEX"])
    
    # File upload section
    with st.sidebar:
        st.header("Configuration ⚙️")
        pdf_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    process_pdfs(pdf_files, embedding_model, pc)
                st.success(f"Processed {len(pdf_files)} documents!")
            else:
                st.warning("Please upload PDF files first")
    
    # Q&A Section
    st.header("Ask a Question ❓")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner("Searching knowledge base..."):
            # Get query embedding
            query_embedding = embedding_model.encode(question).tolist()
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            # Combine context
            context = "\n\n".join([
                f"Source: {match['metadata']['source']} (Page {match['metadata']['page']})\n"
                f"{match['metadata']['text']}" 
                for match in results['matches']
            ])
            
            # Generate answer
            answer = qa_model(question=question, context=context)['answer']
            
            # Display results
            st.subheader("Answer:")
            st.info(answer)
            
            with st.expander("See relevant context"):
                st.markdown(f"**Question:** {question}")
                st.markdown("**Relevant Context:**")
                st.write(context)

if __name__ == "__main__":
    main()