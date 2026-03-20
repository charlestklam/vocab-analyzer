import pandas as pd
from collections import Counter
from wordfreq import top_n_list
import string
import nltk

# Download required NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # For newer nltk versions
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def map_pos_to_spacy_equiv(treebank_tag):
    if treebank_tag.startswith('J'):
        return "ADJ"
    elif treebank_tag.startswith('V'):
        return "VERB"
    elif treebank_tag.startswith('N'):
        return "NOUN" 
    elif treebank_tag.startswith('R'):
        return "ADV"
    else:
        return "OTHER"

def load_bnc_coca_bands():
    """
    Load K-bands from BNC_COCA_lists.json (ground truth).
    Maps word forms (and their lemmas) to their frequency band (K1..K25).
    """
    import os, json
    k_bands = {}
    json_path = os.path.join(os.path.dirname(__file__), 'data', 'BNC_COCA_lists.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            form = item.get('form', '').lower()
            lemma = item.get('lemma', '').lower()
            raw_band = item.get('band', '')
            
            num_part = ''.join(filter(str.isdigit, raw_band))
            if num_part:
                band_label = f"K{num_part}"
            else:
                band_label = "OFF"
                
            if form:
                k_bands[form] = band_label
            if lemma and lemma not in k_bands:
                k_bands[lemma] = band_label
    except Exception as e:
        print(f"Error loading BNC_COCA list: {e}. Falling back to wordfreq.")
        top_words = top_n_list('en', 25000)
        for index, word in enumerate(top_words):
            band = (index // 1000) + 1
            k_bands[word] = f"K{band}"
            
    return k_bands

# Pre-generate the 25k dictionary from ground truth
DEFAULT_K_BANDS = load_bnc_coca_bands()

class LexicalAnalyzer:
    def __init__(self, k_bands=None):
        """
        Initialize the analyzer with a specific word-to-K-band dictionary.
        If none is provided, it defaults to the wordfreq 25K synthesis.
        """
        self.k_bands = k_bands if k_bands is not None else DEFAULT_K_BANDS
    
    def analyze_text(self, text, doc_id="Unknown", level="Unknown"):
        # Tokenize
        raw_tokens = word_tokenize(text)
        
        # Filter for alpha and digits (like is_alpha or is_digit)
        tokens_list = [t for t in raw_tokens if any(c.isalpha() or c.isdigit() for c in t)]
        
        # POS Tag
        pos_tags = nltk.pos_tag(tokens_list)
        
        # Lemmatize
        lemmas_str = []
        for word, tag in pos_tags:
            wn_pos = get_wordnet_pos(tag)
            lemmas_str.append(lemmatizer.lemmatize(word.lower(), wn_pos))
            
        tokens_str = [t.lower() for t in tokens_list]
        
        tokens_total = len(tokens_list)
        types_total = len(set(tokens_str))
        lemmas_total = len(set(lemmas_str))
        
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        
        content_words = []
        function_words = []
        
        pos_data = {
            "NOUN": [],
            "VERB": [],
            "ADJ": [],
            "ADV": [],
        }
        
        for word, tag in pos_tags:
            spacy_pos = map_pos_to_spacy_equiv(tag)
            if spacy_pos in content_pos:
                content_words.append(word)
                pos_data[spacy_pos].append(word)
            else:
                function_words.append(word)
        
        cw_count = len(content_words)
        fw_count = len(function_words)
        ld = cw_count / tokens_total if tokens_total > 0 else 0
        ttr = types_total / tokens_total if tokens_total > 0 else 0
        tt_ratio = tokens_total / types_total if types_total > 0 else 0
        
        res = {
            "ID": doc_id,
            "Level": level,
            "Text": text,
            "Tokens_total": tokens_total,
            "Types_total": types_total,
            "Lemma_total": lemmas_total
        }
        
        for pos, t_list in pos_data.items():
            t_strs = [t.lower() for t in t_list]
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
        families_found = set()
        k_counts = {f"K{i}": 0 for i in range(1, 26)}
        k_counts["OFF"] = 0
        
        for token, lemma in zip(tokens_str, lemmas_str):
            band = self.k_bands.get(token, self.k_bands.get(lemma, "OFF"))
            if band not in k_counts:
                band = "OFF"
            k_counts[band] += 1
            if band != "OFF":
                families_found.add(lemma) 
                
        fam_count = len(families_found)
        res["FAM"] = fam_count
        res["ToPF"] = round(tokens_total / fam_count, 4) if fam_count > 0 else 0
        res["TyPF"] = round(types_total / fam_count, 4) if fam_count > 0 else 0
        
        for k, v in k_counts.items():
            res[k] = v
            
        return res
