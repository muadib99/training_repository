def embed_with_pinecone():
        from dotenv import load_dotenv
        import os
        import json
        import random
        import string
        load_dotenv()
        pinecone_api=os.getenv("pinecone_api")
        gemini_api_key=os.getenv("gemma_gemini_api")

        def generate_random_string(length=14):
            characters = string.ascii_letters + string.digits + "-"
            return ''.join(random.choices(characters, k=length))

        """#create a pinecone index

        """

        # Import the Pinecone library
        from pinecone import Pinecone

        # Initialize a Pinecone client with your API key
        pc = Pinecone(api_key=pinecone_api)

        # Generate random index name and namespace name
        index_name = generate_random_string()
        namespace_name = generate_random_string()

        # Create a dense index with integrated embedding
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model":"llama-text-embed-v2",
                    "field_map":{"text": "chunk_text"}
                }
            )

        """##sample embedding record format

        """

        # records= [
        #     {
        #             "_id": "rec97",
        #             "chunk_text": "He suggested asking the Galactic AC about entropy reversal.",
        #             "category": "document_analysis"
        #         },
        #         {
        #             "_id": "rec98",
        #             "chunk_text": "The AC-contact was small but connected to the Galactic AC.",
        #             "category": "document_analysis"
        #         },
        #         {
        #             "_id": "rec99",
        #             "chunk_text": "MQ-17J wondered about seeing the Galactic AC someday.",
        #             "category": "document_analysis"
        #         }

        #     ]
        import gemini_record
        records= gemini_record.gemini_segement_sentences()
        

        """##upload embeddings to namespace"""

        # Target the index
        dense_index = pc.Index(index_name)

        # # Upsert the records into a namespace
        dense_index.upsert_records(namespace_name, records)

        # Wait for the upserted vectors to be indexed
        import time
        time.sleep(10)

        # View stats for the index
        stats = dense_index.describe_index_stats()
        print(stats)

        from google import genai

        gemini_client = genai.Client(api_key=gemini_api_key)

        """## upload the query findings to gemini"""

        # Define the
        #query = "how does the story end"

        # # Search the dense index
        # results = dense_index.search(
        #     namespace="query-assimov-namespace",
        #     query={
        #         "top_k": 10,
        #         "inputs": {
        #             'text': query
        #         }
        #     }
        # )

        # # Print the results
        # for hit in results['result']['hits']:
        #         print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")

        # hit_texts = []
        # for hit in results['result']['hits']:
        #     text = hit['fields'].get('chunk_text', '')
        #     hit_texts.append(text)
            #print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {text:<50}")

        # combined_text = "\n".join(f"- {t}" for t in hit_texts)


        ##########################--------------------------##############################
        query = "does ac solve the problem of entropy reversal"
        reranked_results = dense_index.search(
            namespace="query-assimov-namespace",
            query={
                "top_k": 10,
                "inputs": {
                    'text': query
                }
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }
        )
        hit_texts = []
        for hit in reranked_results['result']['hits']:
            text = hit['fields'].get('chunk_text', '')
            hit_texts.append(text)
            print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {text:<50}")

        # === Insert summarization using Gemini ===
        # You may choose to limit number of hits if needed
        combined_text = "\n".join(f"- {t}" for t in hit_texts)

        system_prompt = (
            "this is the initial query :{}, and this is the  collection of close answers:{}"
            "if you find that the answers you get dont match with the question asked, just answer on your own"
            "Summarize the following search results in a concise paragraph:\n"
            "if the contents of {} dont contain sufficient answers for {} then answer with, 'not enough data for satisfactory answer'"
            "Only summarize the parts relevant, get straight to the summarization, no need to refer to the structure"
            "answer in a maximum of 3 sentences"
            "make no mention of the search process or the findings process".format(query, combined_text,query,combined_text)
            # "Briefly state the overall finding."
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=[system_prompt]
        )
        return response.txt


