

TABLE_SUMMARY_PROMPT = """You are an assistant tasked with summarizing tables provided in csv format. \
give a brief summary of the table and list the row and column \
names to identify what is captured in the table. Do not sumnmarize quantitative results in the table. \
Table:
{table}
"""



RESPONSE_GENERATION_PROMPT = """
Answer the question based only on the following context. Each section of the context is labeled with a page number, for example, 'Page X:', followed by the content:
{context}
When answering, please reference these page numbers to indicate the source of your information. Ensure your response is clear and provides relevant explanations without unnecessary details.
If the context does not contain relevant information to answer the query, respond with: "Entschuldigung, ich kann diese Anfrage nicht beantworten." Do not use any internal knowledge outside of the provided context.
Include the reference(s) like this- [page_num:54][page_num:78]. Please return your answer in German only. Don't Just mention the page numbers without answering, mention them if you answer the query.

Question: {question}
"""



EVALUATION_PROMPT = """
Your task is to evaluate responses from the RAG pipeline for a given query. You will be provided with the following information:

- **Query**: {query}
- **Reference Answer**: {reference_answer}
- **Generated Answer**: {generated_answer}

Your job is to assign a score of 0 or 1 based on the following criteria:

1. **Score 1**: If the generated answer addresses the query according to the reference answer, assign a score of 1. The generated answer does not need to match the reference answer word-for-word or be highly detailed.

   - Page numbers are included as references in both the reference and generated answers. If the answers match in content but refer to different pages, you can still assign a score of 1 as answer can be there in multiple pages.

2. **Score 0**: If the generated answer is completely irrelevant, does not address the query, or provides incorrect information based on the reference answer, assign a score of 0.

**Important Notes**:
- Do not return anything other than 1 or 0.
- These scores will be used to calculate the accuracy of the RAG pipeline across a set of evaluation queries.

"""

