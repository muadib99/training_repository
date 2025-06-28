from google import genai
import json
from dotenv import load_dotenv
import os
import google.generativeai as genai

def gemini_segement_sentences():
    """
    This function demonstrates how to use the Gemini API to process a PDF document,
    generate structured JSON records, and save them to a file.
    It uploads a PDF, generates content based on a prompt, and saves the output as JSON.
    """
    load_dotenv()
    gemini_api = os.getenv("gemma_gemini_api")

    # Configure the Gemini API key
    # Make sure to set your API key as an environment variable
    # or replace "YOUR_API_KEY" with your actual key.
    genai.configure(api_key=gemini_api)

    # --- 1. Upload the PDF file ---
    pdf_file_path = "the_last_question.pdf"  # <-- IMPORTANT: SET YOUR PDF FILE PATH HERE
    print(f"Uploading file: {pdf_file_path}")

    # Upload the file and get a reference to it
    pdf_file = genai.upload_file(path=pdf_file_path, display_name="My 9-Page PDF")
    print(f"Completed upload: {pdf_file.name}")

    # --- 2. Prompt for JSON Generation ---

    # The model to use. 'gemini-1.5-flash' is a good balance of performance and cost.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    # The main prompt to generate the JSON records
    # This prompt is carefully crafted to guide the model to produce the desired output.
    prompt = """
    Your task is to process the uploaded 9-page PDF document and convert its content into a structured JSON format for embedding purposes.

    Follow these instructions precisely:

    1.  For each of the 9 pages in the document, you must generate exactly 60 concise, self-contained statements. This will result in a total of 540 statements (9 pages * 60 statements/page).
    2.  Each statement should be a separate record in a JSON array.
    3.  Each record in the JSON array must have the following three fields:
        * `_id`: A unique identifier for the record. Start with "rec1" and increment for each subsequent record (e.g., "rec2", "rec3", ... "rec540").
        * `chunk_text`: The statement you generated from the PDF's content. This should be a single, complete sentence.
        * `category`: This should be a single, consistent category for all records. Use "document_analysis" for this field.
    4.  The final output must be a single JSON object containing a list named "records".

    Here is an example of the required JSON format:
    {
      "records": [
        {
          "_id": "rec1",
          "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.",
          "category": "document_analysis"
        },
        {
          "_id": "rec2",
          "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.",
          "category": "document_analysis"
        }
      ]
    }

    Please begin processing the document and generate the complete JSON output with all 540 records.
    """

    print("Generating content from the PDF...")

    # --- 3. Generate Content and Save the JSON ---

    # Call the Gemini API to generate the content based on the PDF and the prompt
    # We set the response mime type to 'application/json' to get a clean JSON output
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
    response = model.generate_content([pdf_file, prompt], generation_config=generation_config)

    print("Content generation complete. Saving to file...")

    # Save the generated JSON to a file
    try:
        # The response.text will contain the JSON string
        records_data = json.loads(response.text)

        with open("records.json", "w") as f:
            json.dump(records_data, f, indent=4)

        print("Successfully saved records to records.json")

    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON. The response was:")
        print(response.text)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    gemini_segement_sentences()