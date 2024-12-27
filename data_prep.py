import os
import pandas as pd
from dotenv import load_dotenv
import aryn_sdk
from aryn_sdk.partition import partition_file, tables_to_pandas
from langchain_core.documents import Document
from typing import List, Tuple
import json
from config import aryn_api_key


def save_documents_to_json(documents, file_path):
    """
    Saves a list of Document objects to a JSON file, converting each Document object into a dictionary.
    This function checks if 'element_type' is present in the metadata and includes it if available.

    Args:
        documents (List[Document]): The list of Document objects to save.
        file_path (str): The path to the JSON file where documents should be saved.
    """
    docs_to_save = []
    for doc in documents:
        doc_dict = {
            'page_content': doc.page_content,
            'page_number': doc.metadata['page_number']
        }
        # Add 'element_type' to the dictionary if it exists in the metadata
        if 'element_type' in doc.metadata:
            doc_dict['element_type'] = doc.metadata['element_type']
        
        docs_to_save.append(doc_dict)
    
    # Write the list of document dictionaries to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(docs_to_save, json_file, indent=4)
        
        

def process_pdf(file_name: str) -> Tuple[List[Document], List[Document]]:
    """
    Processes a PDF file to extract text and tables, filtering out non-essential elements, and creates Document objects
    for both full-page content and individual elements.
    
    Args:
    file_name (str): The path to the PDF file to be processed.

    Returns:
    Tuple[List[Document], List[Document]]: A tuple containing two lists of Document objects. 
    The first list contains documents aggregated by page.
        The second list contains sub-documents for each individual element, including tables and text elements.
    """
    # Load PDF file
    with open(file_name, 'rb') as file:
        partitioned_file = partition_file(file, aryn_api_key, extract_table_structure=True, use_ocr=True)
        
    filtered_file = partitioned_file.copy()

    # Filter out unnecessary elements
    # element_types_to_remove = {'Caption', 'Footnote', 'Image', 'Page-footer', 'Page-header', 'Section-header'}
    element_types_to_remove = {'Footnote', 'Page-footer', 'Page-header','Caption','Image'}
    filtered_file['elements'] = [
        element for element in partitioned_file['elements'] 
        if element['type'] not in element_types_to_remove
    ]
    
    # Extract tables into pandas DataFrames

    pandas, page_texts = tables_to_pandas(filtered_file), {}
    tables = []

    for elt, dataframe in pandas:
        if elt['type'] == 'table':
            tables.append(dataframe)

    table_count = 0
    
    elements = filtered_file['elements']
    # Initialize sub-documents list
    sub_documents = []
    
    # Process each element for documents and sub-documents
    for element in elements:
        page_num = element['properties']['page_number']
        element_type = element['type']

        if element_type == 'table':
            text = tables[table_count].to_csv(index=False)
            table_count += 1
        else:
            text = element['text_representation']

        # Append text to page_texts for document aggregation
        if page_num in page_texts:
            page_texts[page_num] += "\n" + text
        else:
            page_texts[page_num] = text
        
        # Create and store sub-document for each element
        sub_document = Document(
            page_content=text,
            metadata={
                "page_number": page_num,
                "element_type": element_type
            }
        )
        sub_documents.append(sub_document)
        

    # Create Document objects for each page
    documents = [Document(page_content=content, metadata={"page_number": num}) for num, content in page_texts.items()]
    save_documents_to_json(documents, 'documents.json')
    save_documents_to_json(sub_documents, 'sub_documents.json')
    
    return documents, sub_documents


def prepare_context_for_generation(docs, documents):    
    """
    Prepares a text context for document generation by mapping and transforming document content based on page numbers.

    Args:
        docs (List[Document]): A list of document objects for which context needs to be prepared.
        documents (List[Document]): A list of document objects containing the original content to be referenced.

    Returns:
        str: A string composed of processed document content, joined by double newlines.
    """
    # Create a dictionary to map page_number to page_content from documents for quick lookup.
    # This mapping facilitates efficient retrieval of content by page number.
    page_to_content = {doc.metadata['page_number']: doc.page_content for doc in documents}

    # Initialize a list to store the newly processed document contents.
    processed_docs = []
    
    # Iterate through each document in the provided docs list.
    for doc in docs:
        page_num = doc.metadata['page_number']    # Retrieve the page number from the document's metadata.
        # Check if the page number from docs exists in the page_to_content mapping
        if page_num in page_to_content:
            # If exists, create a new document with the corresponding page content from documents
            new_content = f"Page {page_num}: {page_to_content[page_num]}"
        else:
            # If not exists, create a document with 'Unknown Content'
            new_content = f"Page {page_num}: Unknown Content"

        # Append the new document to processed_docs
        processed_docs.append(new_content)
        

    # Join all processed documents into a single string
    return "\n\n".join(processed_docs)

def load_documents_from_json(file_path):
    """
    Loads documents from a JSON file into a list of Document objects.
    This function dynamically adds any available metadata fields to the Document objects.

    Args:
        file_path (str): The path to the JSON file from which to load documents.

    Returns:
        List[Document]: The list of reconstituted Document objects.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    documents = []
    for doc in data:
        metadata = {'page_number': doc['page_number']}
        if 'element_type' in doc:
            metadata['element_type'] = doc['element_type']
        
        document = Document(page_content=doc['page_content'], metadata=metadata)
        documents.append(document)
    
    return documents


def main():
    file_name = 'technical_manual.pdf'
    documents, sub_documents = process_pdf(file_name)
    print(f"Processed {len(documents)} documents.")
    print(f"Processed {len(sub_documents)} sub-documents.")

if __name__ == "__main__":
    main()
