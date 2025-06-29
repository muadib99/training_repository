import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("pinecone_api"))
index_name = "biology-grade-11-general-contents"
dense_index = pc.Index(index_name)

# Define the query and search the dense index
query = "Why are phospholipids?"
results = dense_index.search(
    namespace="contents",
    query={
        "top_k": 5,
        "inputs": {
            'text': query
        }
    }
)

# Initialize the Generative AI client
client = genai.Client(api_key=os.getenv("gemma_gemini_api"))

# Format the search results
formatted_results = "\n".join(
    f"ID: {hit['_id']} | SCORE: {round(hit['_score'], 2)} | PAGE_NUMBER: {hit['fields']['page_number']}"
    "\n"
    f"TEXT_HEADER: {hit['fields']['topic']}"
    "\n"
    f"TEXT_CONTENT: {hit['fields']['chunk_text']}"
    "\n"
    "\n"
    "\n"
    "\n"
    for hit in results['result']['hits']
)
# print(formatted_results,end='\n')
def generate_content(contents):
    """
    Generates a summary of the query response based on the provided contents.
    """
    system_prompt = f"""
    You are a specialized AI assistant. Your sole purpose is to answer the user's query based exclusively on the provided search results. You must adhere to the following instructions without deviation.

    **Instructions:**

    1.  **Analyze the User's Query:** The user wants to know: "{query}"

    2.  **Review the Provided Context:** You are given the following search results, each containing an ID, SCORE, PAGE_NUMBER, TEXT_HEADER, and TEXT_CONTENT.

        ```context
        {contents}
        ```

    3.  **Synthesize the Answer:**
        * Formulate a concise answer to the user's query using *only* the information found in the `TEXT_CONTENT` of the provided results.
        * The answer must be no more than five to seven sentences.
        * Do not invent, infer, or use any information outside of the provided context.

    4.  **Cite Your Sources:**
        * After the answer, list the sources you used.
        * For each source, you must include its `ID`, `SCORE`, and `PAGE_NUMBER`.
        * Format each citation exactly as: `Source: \n ID: [ID], SCORE: [SCORE], PAGE_NUMBER: [PAGE_NUMBER]`

    **Output Mandate:**

    * Your entire output must consist of two parts ONLY: the synthesized answer first, followed by the list of source citations.
    * DO NOT add any introductory phrases, greetings, apologies, or concluding remarks.
    * DO NOT use any formatting other than what is specified.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=system_prompt
    )
    print(response.text)

# Generate and print the content
generate_content(formatted_results)