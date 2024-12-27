from config import llm
from prompts import TABLE_SUMMARY_PROMPT
import json


def generate_and_save_table_summaries(tables, filename = 'table_summary'):
    """
    Generates summaries for a list of tables using a language model and saves the summaries to a JSON file.

    Args:
        tables (list): A list of table data, each entry formatted as needed by the TABLE_SUMMARY_PROMPT.
        filename (str, optional): The name of the file to save the summaries. Defaults to 'table_summary'.

    Returns:
        list: A list of generated text summaries for each table.
    """
    
    # Notify the start of summary generation process.
    print('started generating summaries')
    text_summaries = []
    for table in tables:
        # Format the prompt with the current table's data.
        filled_prompt = TABLE_SUMMARY_PROMPT.format(table=table)
        # Invoke the language model with the formatted prompt.
        response = llm.invoke(filled_prompt)
        # Append the content of the response to the text_summaries list.
        text_summaries.append(response.content)
    
    # Save summaries to JSON
    with open(filename, 'w') as file:
        json.dump(text_summaries, file)
        
    # Notify the end of the summary generation process.
    print('finished generating summaries')
    return text_summaries




def load_table_summaries(filename):
    """
    Loads and returns table summaries from a specified JSON file.

    Args:
        filename (str): The name of the JSON file from which to load the summaries.

    Returns:
        list: A list of loaded table summaries.
    """
    with open(filename, 'r') as file:
        summaries = json.load(file)
    return summaries