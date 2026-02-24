import spacy
import pandas as pd
from collections import Counter
from wordfreq import top_n_list
# Load the core English model from spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def generate_k_bands_dictionary():
    """
    Synthesize a K1-K25 mapping using the top 25,000 English words from wordfreq.
    Maps lemmas to their frequency band (K1..K25).
    """
    k_bands = {}
    top_words = top_n_list('en', 25000)
    for index, word in enumerate(top_words):
        band = (index // 1000) + 1
        k_bands[word] = f"K{band}"
    return k_bands

# Pre-generate the 25k dictionary
DEFAULT_K_BANDS = generate_k_bands_dictionary()

class LexicalAnalyzer:
    def __init__(self, k_bands=None):
        """
        Initialize the analyzer with a specific word-to-K-band dictionary.
        If none is provided, it defaults to the wordfreq 25K synthesis.
        """
        self.k_bands = k_bands if k_bands is not None else DEFAULT_K_BANDS
    
    def analyze_text(self, text, doc_id="Unknown", level="Unknown"):
        doc = nlp(text)
        
        # 1. Basic Counts
        # Exclude pure punctuation and spaces from token count if desired, but typically 
        # vocabulary profile tools count alphabetic/numeric tokens.
        tokens_list = [t for t in doc if t.is_alpha or t.is_digit]
        tokens_str = [t.text.lower() for t in tokens_list]
        lemmas_str = [t.lemma_.lower() for t in tokens_list]
        
        tokens_total = len(tokens_list)
        types_total = len(set(tokens_str))
        
        # Function Words vs Content Words
        content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
        content_words = [t for t in tokens_list if t.pos_ in content_pos]
        function_words = [t for t in tokens_list if t.pos_ not in content_pos]
        
        cw_count = len(content_words)
        fw_count = len(function_words)
        ld = cw_count / tokens_total if tokens_total > 0 else 0
        ttr = types_total / tokens_total if tokens_total > 0 else 0
        tt_ratio = tokens_total / types_total if types_total > 0 else 0
        
        # 2. Part of Speech Grouping
        pos_data = {
            "NOUN": [t for t in tokens_list if t.pos_ in ("NOUN", "PROPN")],
            "VERB": [t for t in tokens_list if t.pos_ == "VERB"],
            "ADJ": [t for t in tokens_list if t.pos_ == "ADJ"],
            "ADV": [t for t in tokens_list if t.pos_ == "ADV"],
        }
        
        res = {
            "ID": doc_id,
            "Level": level,
            "Text": text,
            "Tokens_total": tokens_total,
            "Types_total": types_total
        }
        
        for pos, t_list in pos_data.items():
            t_strs = [t.text.lower() for t in t_list]
            t_types = set(t_strs)
            
            res[f"{pos}_tokens"] = len(t_list)
            res[f"{pos}_types"] = len(t_types)
            res[f"{pos}S_types" if pos != "ADJ" and pos != "ADV" else f"{pos}s_types"] = ", ".join(sorted(t_types))
            
        res.update({
            "TTR": round(ttr, 4),
            "T/T": round(tt_ratio, 4),
            "LD": round(ld, 4),
            "CW": cw_count,
            "FW": fw_count,
        })
        
        # 3. K-Band Frequency Distribution (Lexical Profiling)
        # BNC/COCA lists typically map word families (represented by the lemma).
        families_found = set()
        k_counts = {f"K{i}": 0 for i in range(1, 26)}
        k_counts["OFF"] = 0
        
        for lemma in lemmas_str:
            band = self.k_bands.get(lemma, "OFF")
            k_counts[band] += 1
            if band != "OFF":
                families_found.add(lemma) # Approximate word family matching for now via lemmas
                
        fam_count = len(families_found)
        res["FAM"] = fam_count
        res["ToPF"] = round(tokens_total / fam_count, 4) if fam_count > 0 else 0
        res["TyPF"] = round(types_total / fam_count, 4) if fam_count > 0 else 0
        
        for k, v in k_counts.items():
            res[k] = v
            
        return res
