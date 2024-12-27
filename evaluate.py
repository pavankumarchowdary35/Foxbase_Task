import pandas as pd
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from data_prep import prepare_context_for_generation
from config import llm, text_embeddings
from prompts import RESPONSE_GENERATION_PROMPT, EVALUATION_PROMPT
import os
from data_prep import load_documents_from_json
import json
from langchain_core.documents import Document
import streamlit as st
from config import vector_store_path, k_value

        
def evaluate_queries(csv_file_path, vector_store_path, text_embeddings, RESPONSE_GENERATION_PROMPT, EVALUATION_PROMPT, output_file_path):
    """
    Evaluates queries from a CSV file using a RAG pipeline and LLM for response generation and scoring.

    Parameters:
        csv_file_path (str): Path to the input CSV file containing the queries and reference answers.
        vector_store_path (str): Path to the vector store directory for Chroma.
        text_embeddings (function): Embedding function for vector database initialization.
        RESPONSE_GENERATION_PROMPT (str): Prompt template for generating responses using the LLM.
        EVALUATION_PROMPT (str): Prompt template for evaluating the generated responses.
        output_file_path (str): Path to save the CSV file containing evaluation results.

    Returns:
        None

    This function processes each query in the input CSV file, generates responses using the RAG pipeline, evaluates the responses, 
    and saves the results with generated answers and evaluation scores in the output CSV file.
    """
    import pandas as pd

    # Load the CSV file
    data = pd.read_csv(csv_file_path)
    data.columns = data.columns.str.strip()

    # Initialize your LLM pipeline and retrievers
    vectordb = Chroma(persist_directory=vector_store_path, embedding_function=text_embeddings)
    
    sub_documents = load_documents_from_json('sub_documents.json')
    docs_retriever = vectordb.as_retriever(search_kwargs={"k": 10}) #search_type="mmr",
    bm25_retriever = BM25Retriever.from_documents(sub_documents)
    ensemble_retriever = EnsembleRetriever(retrievers=[docs_retriever, bm25_retriever], weights=[0.5, 0.5])
    documents = load_documents_from_json('documents.json')

    # Add columns for generated answer and evaluation score
    data['Generated Answer'] = ""
    data['Eva_Score'] = 0

    # Process each query
    for index, row in data.iterrows():
        try:
            print(f"Processing query {index + 1}/{len(data)}: {row['Frage']}")
            query = row['Frage']
            reference_answer = row['Antwort']
            
            # Generate answer
            k_value = k_value
            generated_answer = None
            while k_value > 0:
                try:
                    docs_retriever.search_kwargs["k"] = k_value
                    retrieved_docs = ensemble_retriever.invoke(query)
                    context = prepare_context_for_generation(retrieved_docs, documents)
                    filled_prompt = RESPONSE_GENERATION_PROMPT.format(context=context, question=query)
                    response = llm.invoke(filled_prompt)
                    generated_answer = response.content.strip()
                    break
                except Exception as e:
                    if k_value > 2:
                        k_value -= 2
                    else:
                        generated_answer = "Error generating answer."
                        raise e
            
            # Store the generated answer
            data.at[index, 'Generated Answer'] = generated_answer
            
            # Evaluate the generated answer
            eval_prompt = EVALUATION_PROMPT.format(query=query, reference_answer=reference_answer, generated_answer=generated_answer)
            try:
                eval_response = llm.invoke(eval_prompt)
                eval_score = int(eval_response.content.strip())
            except Exception as e:
                print(f"Error during evaluation: {e}")
                eval_score = 0  # Default to 0 if evaluation fails
            
            # Store the evaluation score
            data.at[index, 'Eva_Score'] = eval_score
        
        except Exception as e:
            print(f"Failed to process query at index {index}: {e}")
            data.at[index, 'Generated Answer'] = "Error processing query."
            data.at[index, 'Eva_Score'] = 0

    # Save the results to a new CSV file
    data.to_csv(output_file_path, index=False)

    print(f"Evaluation completed. Results saved to {output_file_path}.")


def main():
    """
    Main function to execute the query evaluation process.
    """

    evaluate_queries(
        csv_file_path='questions_answers.csv',
        vector_store_path=vector_store_path,
        text_embeddings=text_embeddings,
        RESPONSE_GENERATION_PROMPT=RESPONSE_GENERATION_PROMPT,
        EVALUATION_PROMPT=EVALUATION_PROMPT,
        output_file_path='evaluation_results.csv'
    )

if __name__ == "__main__":
    main()




