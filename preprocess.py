import json
import glob
import re
import os
import pandas as pd
from analyzer import LexicalAnalyzer

# Initialize analyzer
analyzer = LexicalAnalyzer()

# 1. Process any .json file in data/ whose name ends with a number (e.g. batch_50, generated_40)
json_files = sorted(glob.glob('data/*.json'))
for json_file in json_files:
    basename = os.path.splitext(os.path.basename(json_file))[0]
    # Only process files whose name ends with _<identifier> (e.g. batch_50, generated_40, openalex_5k)
    if not re.search(r'_\w+$', basename):
        continue
    # Skip files that are already results
    if basename.endswith('_results'):
        continue

    results_file = f'data/{basename}_results.json'

    # Skip if results already exist
    if os.path.exists(results_file):
        print(f"Skipping {basename} (results already exist)")
        continue

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Pre-calculating {basename} data ({len(data)} records)...")
    results = []
    for item in data:
        doc_id = str(item.get("ID", "Unknown"))
        level = str(item.get("Level", "Unknown"))
        text = str(item.get("Text", ""))
        if text:
            res = analyzer.analyze_text(text, doc_id=doc_id, level=level)
            results.append(res)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  -> Saved {len(results)} results to {results_file}")

# 2. Process openalex_ai_criticality_filtered.jsonl (if it exists)
jsonl_file = 'data/openalex_ai_criticality_filtered.jsonl'
if os.path.exists(jsonl_file):
    print("Pre-calculating openalex data...")
    openalex_data = []
    openalex_results = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
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
