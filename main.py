from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from data_prep import prepare_context_for_generation
from config import llm, text_embeddings
from prompts import RESPONSE_GENERATION_PROMPT
import os
from data_prep import load_documents_from_json
import json
from langchain_core.documents import Document
import streamlit as st
from config import vector_store_path, k_value


st.title("RAG System")
st.write("Submit your query below.")

# Input field for query
query = st.text_input("Enter your query:", "")

if st.button("Get Answer"):
    if query.strip():
        try:
            # Load VectorDB
            vectordb = Chroma(persist_directory=vector_store_path, embedding_function=text_embeddings)
            sub_documents = load_documents_from_json('sub_documents.json')
            documents = load_documents_from_json('documents.json')

            # Define retrievers
            docs_retriever = vectordb.as_retriever()
            bm25_retriever = BM25Retriever.from_documents(sub_documents)
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, docs_retriever], weights=[0.5, 0.5])

            # Handle context limit with dynamic k_value adjustment
            response_content = ""

            while k_value > 0:
                try:
                    docs_retriever.search_kwargs["k"] = k_value
                    retrieved_docs = ensemble_retriever.invoke(query)

                    # Prepare context
                    context = prepare_context_for_generation(retrieved_docs, documents)

                    # Generate response
                    filled_prompt = RESPONSE_GENERATION_PROMPT.format(context=context, question=query)
                    response = llm.invoke(filled_prompt)
                    response_content = response.content
                    break
                except Exception as e:
                    if k_value > 2:
                        k_value -= 2
                    else:
                        response_content = "Unable to generate response due to context limitations."
                        raise e

            # Display the response
            if response_content:
                st.subheader("Generated Response:")
                st.write(response_content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to get a response.")

st.write("\n\n*Note: This system retrieves relevant chunks and dynamically adjusts context size to handle LLM limitations and generate responses based on the provided query.")