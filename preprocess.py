import json
import pandas as pd
from analyzer import LexicalAnalyzer

# Initialize analyzer
analyzer = LexicalAnalyzer()

# 1. Process batch_50.json (already formatted)
with open('data/batch_50.json', 'r', encoding='utf-8') as f:
    batch_data = json.load(f)

print(f"Pre-calculating batch_50 data ({len(batch_data)} records)...")
batch_results = []
for item in batch_data:
    doc_id = str(item.get("ID", "Unknown"))
    level = str(item.get("Level", "Unknown"))
    text = str(item.get("Text", ""))
    if text:
        res = analyzer.analyze_text(text, doc_id=doc_id, level=level)
        batch_results.append(res)

with open('data/batch_50_results.json', 'w', encoding='utf-8') as f:
    json.dump(batch_results, f, ensure_ascii=False, indent=2)

# 2. Process openalex_ai_criticality_filtered.jsonl
print("Pre-calculating openalex data...")
openalex_data = []
openalex_results = []

with open('data/openalex_ai_criticality_filtered.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        abstract = row.get('abstract', '')
        if abstract and len(abstract) > 10:  # Filter out empty or very short abstracts
            doc_id = str(row.get('openalex_id', 'Unknown'))
            level = str(row.get('category', 'Unknown')) # Use category as level
            
            # Format like batch_50 just in case we need the raw data
            openalex_data.append({
                "ID": doc_id,
                "Level": level,
                "Text": abstract
            })
            
            # Pre-calculate
            res = analyzer.analyze_text(abstract, doc_id=doc_id, level=level)
            openalex_results.append(res)

print(f"Processed {len(openalex_results)} OpenAlex records.")

with open('data/openalex_data.json', 'w', encoding='utf-8') as f:
    json.dump(openalex_data, f, ensure_ascii=False, indent=2)

with open('data/openalex_results.json', 'w', encoding='utf-8') as f:
    json.dump(openalex_results, f, ensure_ascii=False, indent=2)

print("Pre-calculation complete. Files saved to data/ directory.")
