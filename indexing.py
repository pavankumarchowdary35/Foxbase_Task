from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from data_prep import process_pdf
from generate_table_summary import generate_and_save_table_summaries, load_table_summaries
from config import llm, text_embeddings
import os
from config import manual_path, summary_filename, vector_store_path



def prepare_documents_and_vector_store(pdf_path: str, summary_filename: str):
    """
    Processes a PDF file, generates or loads table summaries, and creates a vector store from the processed documents,
    without returning any object.

    Args:
        pdf_path (str): Path to the PDF file to be processed.
        summary_filename (str): File name for storing/loading the table summaries.
    """
    documents, sub_documents = process_pdf(pdf_path)
    print(f"Processed {len(sub_documents)} documents.")

    table_texts = [doc.page_content for doc in sub_documents if doc.metadata['element_type'] == 'table']

    if os.path.exists(summary_filename):
        table_summaries = load_table_summaries(summary_filename)
    else:
        table_summaries = generate_and_save_table_summaries(table_texts, summary_filename)

    print('Length of summaries is', len(table_summaries))

    summary_iter = iter(table_summaries)

    for sub_document in sub_documents:
        if sub_document.metadata['element_type'] == 'table':
            sub_document.page_content = next(summary_iter)

    # Check if the vector store directory does not exist, then create the vector store
    if not os.path.exists(vector_store_path):
        vector_store = Chroma.from_documents(documents=sub_documents, persist_directory=vector_store_path, embedding=text_embeddings)
        print(f"Vector store created at {vector_store_path}")
    else:
        print("Vector store directory already exists. Skipping creation.")



if __name__ == '__main__':
    # Provide the path to the PDF file to process.
    pdf_path = manual_path
    prepare_documents_and_vector_store(pdf_path, summary_filename)