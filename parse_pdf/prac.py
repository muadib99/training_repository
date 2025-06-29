import json

# Load the JSON data
with open('keyword_definitions_3_renumbered.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Subtract 4 from each page_number
for entry in data:
    if 'page_number' in entry and isinstance(entry['page_number'], int):
        entry['page_number'] = entry['page_number'] - 4

# Save the updated data
with open('keyword_definitions_3_renumbered.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)