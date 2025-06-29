import google.generativeai as genai
import pdfplumber
import os
import json
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# This function remains the same
def extract_text_from_pages(pages):
    """Extracts text from a list of pdfplumber page objects."""
    if not pages:
        return ""
    
    full_text = ""
    for page in pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

# This function and its detailed prompt remain the same
def extract_headers_and_text_with_gemini(api_key, text_chunk):
    """
    Sends a text chunk to the Gemini API to identify numbered headers and extract their corresponding text.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are an expert data extractor specializing in academic textbooks.
        Your task is to meticulously parse the provided text from a textbook and extract content ONLY from sections with numbered headers.

        **Instructions:**
        1.  **Identify Headers:** Look for headers that start with a number, a period, and another number, like "1.1", "1.2", "1.3", etc. and headers that ask questions like "What is science?" or "what is the scientific Method?" and other headers that dont seem to be activities or figures[images] but rather descriptive statements like "validity" or "The methods of science" and dont extract text where its startes with "by the end of this section....", and it goes on to state the objective.
        2.  **Extract Content:** For each numbered header you find, extract all the text that follows it, right up until the next numbered header begins.
        3.  **Strict Exclusion Rules:**
            * **DO NOT** extract content from general headers such as "UNIT 1", "Activity", "KEY WORDS", "Review questions","End of unit questions","Contents", "Learning competencies", or "Figure".
            * **DO NOT** include any text from sidebars, key word definition boxes, activity instructions, or image captions.
            * **DO NOT** include page numbers, footers, or running headers from the original document.
            * **IGNORE** any introductory paragraphs that appear before the very first numbered header (e.g., before "1.1 The methods of science").

        **Output Format:**
        * Your output **MUST** be a single, valid JSON list of objects.
        * Each object **MUST** contain the following key-value pairs:
            1.  `"_id"`: goes sequentially from one on eg, "rec_1","rec_2","rec_3","rec_4","rec_5","rec_6","rec_7","rec_8".
            1.  `"topic"`: should be the header you find that encloses the text. the header should not be the following: "UNIT 1", "Activity","Review questions","End of unit questions", "KEY WORDS", "Contents", "Learning competencies", or "Figure". headers should look like :that start with a number, a period, and another number, like "1.1", "1.2", "1.3", etc. and headers that ask questions like "What is science?" or "what is the scientific Method?" and other headers that dont seem to be activities or figures[images] but rather descriptive statements like "validity" or "The methods of science" and dont extract text where its startes with "by the end of this section....", and it goes on to state the objective.
            2.  `"chunk_text"`: A string containing all the text content under that header.
            3.  `"page_number"`: the page number where the key word and definition were found, formatted as an integer.

        **Example of Desired Output:**
        ```json
        [
          {{
            "_id":"rec_1".
            "topic": "1.1 The methods of science",
            "chunk_text": "is as a way of looking at and thinking about natural events..."
            "page_number": 1.
          }}
        ]
        ```

        If the text chunk contains no structured content that matches these rules, return an empty JSON list: `[]`.

        **Here is the text chunk to analyze:**
        ---
        {text_chunk}
        ---
        """
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        if response.candidates and response.candidates[0].finish_reason.name != "STOP":
            print(f"Warning: Content generation stopped for reason: {response.candidates[0].finish_reason.name}")
            if response.prompt_feedback.block_reason:
                return None, f"API call blocked by safety settings: {response.prompt_feedback.block_reason.name}"

        cleaned_response = response.text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        
        return cleaned_response, None
    except Exception as e:
        return None, f"An error occurred with the Gemini API: {e}"

# This function remains the same
def save_data_to_json(data_list, output_path):
    """Saves the list of dictionaries to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4)
        return True, None
    except Exception as e:
        return False, f"Error saving to JSON file: {e}"


# --- MODIFIED main() FUNCTION ---
# --- FULLY REVISED main() FUNCTION WITH BATCHING ---

def main():
    """Main function to run the book content extractor using a dictionary and batch processing."""
    
    load_dotenv()
    YOUR_API_KEY = os.getenv("gemma_gemini_api") 
    INPUT_PDF_PATH = "/workspaces/training_repository/parse_pdf/Grade-11-Biology-Textbook.pdf"
    OUTPUT_JSON_PATH = "structured_biology_content_2.json"
    
    # --- NEW: Define a batch size ---
    # This is the number of pages to process in each API call. Tune if needed.
    BATCH_SIZE = 10

    PAGE_CHUNKS = {
        "Unit 1": {"start": 4, "end": 42},
        "Unit 2": {"start": 46, "end": 78},
        "Unit 3": {"start": 83, "end": 105},
        "Unit 4": {"start": 116, "end": 150},
        "Unit 5": {"start": 156, "end": 194},      
    }

    if not YOUR_API_KEY:
        print("Error: Gemini API key not found.")
        return

    if not os.path.exists(INPUT_PDF_PATH):
        print(f"Error: PDF file not found at '{INPUT_PDF_PATH}'")
        return

    all_structured_data = []

    print("--- Starting PDF Processing by Defined Chunks (in Batches) ---")
    try:
        with pdfplumber.open(INPUT_PDF_PATH) as pdf:
            num_total_pages = len(pdf.pages)
            print(f"Total pages in document: {num_total_pages}. Processing in batches of {BATCH_SIZE} pages.")
            
            # Loop over the main Units
            for chunk_name, pages in PAGE_CHUNKS.items():
                start_page = pages['start']
                end_page = pages['end']

                print(f"\n--- Processing '{chunk_name}' (Pages {start_page + 1} to {end_page}) ---")
                
                if start_page >= num_total_pages:
                    print(f"Start page {start_page + 1} is out of bounds. Skipping unit.")
                    continue
                
                # --- NEW: Loop over the unit in smaller batches ---
                for i in range(start_page, end_page, BATCH_SIZE):
                    batch_start = i
                    batch_end = min(i + BATCH_SIZE, end_page) # Ensure we don't go past the unit's end page
                    
                    print(f"  -> Processing batch: Pages {batch_start + 1} to {batch_end}")

                    page_batch = pdf.pages[batch_start:batch_end]
                    
                    if not page_batch:
                        continue

                    batch_text = extract_text_from_pages(page_batch)
                    
                    if not batch_text.strip():
                        print("     No text extracted from this batch. Skipping.")
                        continue
                    
                    print(f"     Extracted {len(batch_text)} characters. Sending to Gemini...")
                    
                    structured_data_str, error = extract_headers_and_text_with_gemini(YOUR_API_KEY, batch_text)
                    
                    if error:
                        print(f"     Error processing batch: {error}")
                        continue 

                    print("     ...structured data received.")
                    
                    if structured_data_str:
                        try:
                            parsed_data = json.loads(structured_data_str)
                            if isinstance(parsed_data, list):
                                all_structured_data.extend(parsed_data)
                            else:
                                print("     Warning: API did not return a list. Response skipped.")
                        except json.JSONDecodeError:
                            print(f"     Warning: Could not decode JSON from API. Response was: {structured_data_str}")

    except Exception as e:
        print(f"An error occurred while opening or reading the PDF: {e}")
        return

    print(f"\n--- All chunks processed. Found a total of {len(all_structured_data)} sections. ---")
    
    # --- IMPORTANT: Re-apply sequential IDs after all data is collected ---
    final_data_with_ids = []
    for i, record in enumerate(all_structured_data):
        record['_id'] = f'rec_{i + 1}'
        final_data_with_ids.append(record)

    print(f"Saving final structured data to '{OUTPUT_JSON_PATH}'...")
    
    success, error = save_data_to_json(final_data_with_ids, OUTPUT_JSON_PATH)
    
    if error:
        print(error)
    else:
        print(f"\nAll done! Your structured content has been saved to {OUTPUT_JSON_PATH}")
        print("\n--- FINAL OUTPUT PREVIEW (first 2 items) ---")
        print(json.dumps(final_data_with_ids[:2], indent=4))
        print("-" * 35)


if __name__ == "__main__":
    main()

