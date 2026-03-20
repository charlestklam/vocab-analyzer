import streamlit as st
import pandas as pd
import json
import io
import os
import glob
import re
import altair as alt
from collections import Counter
from itertools import islice
from analyzer import LexicalAnalyzer

st.set_page_config(page_title="Vocabulary Analyzer", layout="wide")

# Load external CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.sidebar.title("Vocabulary Analyzer")

# st.sidebar.header("Options")
# file_type = st.sidebar.radio("Upload Format", ["Text Files (.txt)", "JSON File (.json)"], help="Choose whether you are uploading multiple TXT files or a structured JSON dump.")

# uploaded_files = st.sidebar.file_uploader(
#     "Choose files", 
#     type=["txt"] if "Text" in file_type else ["json"], 
#     accept_multiple_files=True if "Text" in file_type else False
# )

@st.cache_resource
def load_analyzer():
    # Cache the NLP model and dictionary generation to prevent reloading per run
    return LexicalAnalyzer()

@st.cache_data(show_spinner="Extracting POS tags (this might take a minute on large datasets)...")
def get_pos_distribution(texts, use_lemma=True):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from collections import Counter
    
    lemmatizer = WordNetLemmatizer()
    nouns, verbs, adjs, advs = Counter(), Counter(), Counter(), Counter()
    
    for text in texts:
        if not text:
            continue
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if any(c.isalpha() or c.isdigit() for c in t)]
        tags = nltk.pos_tag(tokens)
        
        for w, tag in tags:
            w_lower = w.lower()
            if tag.startswith('N'):
                target = lemmatizer.lemmatize(w_lower, nltk.corpus.wordnet.NOUN) if use_lemma else w_lower
                nouns[target] += 1
            elif tag.startswith('V'):
                target = lemmatizer.lemmatize(w_lower, nltk.corpus.wordnet.VERB) if use_lemma else w_lower
                verbs[target] += 1
            elif tag.startswith('J'):
                target = lemmatizer.lemmatize(w_lower, nltk.corpus.wordnet.ADJ) if use_lemma else w_lower
                adjs[target] += 1
            elif tag.startswith('R'):
                target = lemmatizer.lemmatize(w_lower, nltk.corpus.wordnet.ADV) if use_lemma else w_lower
                advs[target] += 1
                
    return nouns, verbs, adjs, advs

analyzer = load_analyzer()

if "results" not in st.session_state:
    st.session_state["results"] = []

if "active_view" not in st.session_state:
    st.session_state["active_view"] = "Descriptive Statistics"

# Discover available datasets: any data/*.json that has a matching *_results.json
st.sidebar.header("Datasets")
results_files = sorted(glob.glob('data/*_results.json'))
for results_path in results_files:
    basename = os.path.basename(results_path).replace('_results.json', '')
    data_path = f'data/{basename}.json'
    if not os.path.exists(data_path):
        continue
        
    # User-friendly display names
    display_mapping = {
        "batch_50": "Grade 7 Writing",
        "openalex_5k": "AI Criticality Abstracts"
    }
    display_name = display_mapping.get(basename, basename)
    
    # Count records for the label
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            count = len(json.load(f))
    except Exception:
        count = '?'
        
    label = f"{display_name} (n={count})"
    if st.sidebar.button(label, key=f"dataset_{basename}"):
        st.session_state["current_dataset"] = basename
        with st.spinner(f"Loading {display_name}..."):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    st.session_state["results"] = json.load(f)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

if "current_dataset" not in st.session_state:
    st.session_state["current_dataset"] = None

st.sidebar.header("Filters")
col1, col2 = st.sidebar.columns(2)
min_words = col1.number_input("Min Words", min_value=0, value=0, step=10)
max_words = col2.number_input("Max Words", min_value=0, value=1000, step=10)

available_groups = []
if st.session_state["results"]:
    available_groups = sorted(list(set(str(r.get("Level", "Unknown")) for r in st.session_state["results"])))
selected_groups = st.sidebar.multiselect("Filter by Group(s)", options=available_groups, default=available_groups)

if False and st.button("Run Analysis on Uploaded Data"):  # Hidden for now
    new_results = []
    with st.spinner("Processing..."):
        if not uploaded_files:
            st.warning("Please upload at least one file to proceed.")
        else:
            if "Text" in file_type:
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    filename = file.name.replace('.txt', '')
                    res = analyzer.analyze_text(content, doc_id=filename, level="N/A")
                    new_results.append(res)
            else:
                content = uploaded_files.read().decode("utf-8")
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            doc_id = str(item.get("id", item.get("ID", "Unknown")))
                            level = str(item.get("level", item.get("Level", "Unknown")))
                            text = str(item.get("text", item.get("Text", "")))
                            if text:
                                res = analyzer.analyze_text(text, doc_id=doc_id, level=level)
                                new_results.append(res)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please ensure the file is properly structured.")
    
    if new_results:
        st.session_state["results"] = new_results

results = []
if not st.session_state["results"]:
    st.title("Vocabulary Analyzer")
    st.markdown(
        "Welcome to the Vocabulary Analyzer! This tool allows you to explore the lexical frequency profiles, "
        "syntactic complexity, and part-of-speech distributions of textual data. You can also explore the text with corpus tools "
        "like concordancing and n-gram analysis."
    )
    st.info("**Please click on one of the datasets in the left sidebar to start!**")

if st.session_state["results"]:
    for r in st.session_state["results"]:
        if min_words <= r.get("Tokens_total", 0) <= max_words:
            if not available_groups or str(r.get("Level", "Unknown")) in selected_groups:
                results.append(r)
            
    if not results:
        st.warning("No texts match the current filter criteria.")

    if results:
        df = pd.DataFrame(results)
        st.success(f"Analysis complete for {len(results)} texts.")

        # Tab selector with state persistence
        view_options = ["Descriptive Statistics", "Visualization", "Corpus Tools"]
        st.session_state["active_view"] = st.radio(
            "Select view",
            view_options,
            index=view_options.index(st.session_state["active_view"]),
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("---")

        if st.session_state["active_view"] == "Descriptive Statistics":
            st.subheader("Corpus Summary")
            
            num_texts = len(df)
            total_tokens = int(df['Tokens_total'].sum())
            
            # Quick corpus-level type calculation using regex directly on texts
            if 'Text' in df.columns:
                all_text = " ".join(df['Text'].dropna().astype(str))
                # similar generic tokenization to catch words/numbers
                corpus_types = len(set(t.lower() for t in re.findall(r"[a-zA-Z0-9]+", all_text)))
            else:
                corpus_types = "N/A"
                
            avg_tokens = round(df['Tokens_total'].mean(), 1)
            min_len = int(df['Tokens_total'].min())
            max_len = int(df['Tokens_total'].max())
            
            c1, c2 = st.columns(2)
            c1.metric("Number of texts", f"{num_texts:,}")
            c2.metric("Total tokens", f"{total_tokens:,}")
            c1.metric("Total types", f"{corpus_types:,}" if isinstance(corpus_types, int) else corpus_types, help="Unique words across the entire filtered corpus")
            c2.metric("Tokens per text", f"{avg_tokens:,}")
            c1.metric("Range of length", f"{min_len:,} - {max_len:,}")
            
            st.markdown("---")
            st.markdown("**First 50 Documents:**")
            st.dataframe(df.head(50))
            
            # Allow downloading
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="Vocabulary_Analysis_Results.csv",
                mime="text/csv"
            )

        elif st.session_state["active_view"] == "Visualization":
            st.subheader("Lexical Frequency Profile (K-Bands)")
            
            col_s1, col_s2 = st.columns(2)
            individual_n = col_s1.slider("First N bands to report individually", min_value=0, max_value=25, value=5)
            group_size = col_s2.slider("Group remaining bands by size", min_value=1, max_value=25, value=5)
            
            band_groups = []
            tier_names = []
            
            # Individual bands
            for i in range(1, min(individual_n + 1, 26)):
                band_groups.append([f"K{i}"])
                tier_names.append(f"K{i}")
            
            # Grouped bands
            start = individual_n + 1
            while start <= 25:
                end = min(start + group_size - 1, 25)
                band_groups.append([f"K{i}" for i in range(start, end + 1)])
                if start == end:
                    tier_names.append(f"K{start}")
                else:
                    tier_names.append(f"K{start}-K{end}")
                start = end + 1
            
            # Always include OFF-list words
            band_groups.append(["OFF"])
            tier_names.append("OFF")
            
            tier_sums = {t: 0 for t in tier_names}
            total_corpus_words = 0
            
            for _, row in df.iterrows():
                total_doc_words = sum(row.get(f'K{i}', 0) for i in range(1, 26)) + row.get('OFF', 0)
                total_corpus_words += total_doc_words
                for i, grp in enumerate(band_groups):
                    tier_sums[tier_names[i]] += sum(row.get(k, 0) for k in grp)
            
            if total_corpus_words > 0:
                tier_agg_data = {
                    "Band": tier_names,
                    "Percentage": [tier_sums[t] / total_corpus_words for t in tier_names]
                }
                tier_df = pd.DataFrame(tier_agg_data)
                
                max_perc = tier_df['Percentage'].max()
                perc_domain_max = max_perc * 1.25 if pd.notnull(max_perc) and max_perc > 0 else 1
                
                base_lfp = alt.Chart(tier_df).encode(
                    y=alt.Y('Band:O', sort=tier_names, title='Band'),
                    x=alt.X('Percentage:Q', axis=alt.Axis(format='%', title='Percentage in Corpus'), scale=alt.Scale(domain=[0, perc_domain_max])),
                    tooltip=[alt.Tooltip('Band:O', title='Band'), alt.Tooltip('Percentage:Q', format='.2%', title='Percentage')]
                )
                
                bars_lfp = base_lfp.mark_bar()
                text_lfp = base_lfp.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3
                ).encode(
                    text=alt.Text('Percentage:Q', format='.1%')
                )
                
                chart_lfp = (bars_lfp + text_lfp).properties(
                    title='Lexical Frequency Profile of the Corpus'
                )
                st.altair_chart(chart_lfp, use_container_width=True)
            else:
                st.info("No valid text data to display the profile.")

            st.divider()

            st.subheader("Syntactic Complexity & Lexical Richness")
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot of Lexical Density
                chart_ld = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x=alt.X('LD:Q', title='Lexical Density (Content Words / Total)'),
                    y=alt.Y('Level:N', title='Group') if 'Level' in df.columns else alt.datum('Corpus')
                ).properties(title='Lexical Density Distribution')
                st.altair_chart(chart_ld, use_container_width=True)
                
            with col2:
                # Boxplot of TTR
                chart_ttr = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x=alt.X('TTR:Q', title='Type-Token Ratio'),
                    y=alt.Y('Level:N', title='Group') if 'Level' in df.columns else alt.datum('Corpus')
                ).properties(title='TTR Distribution')
                st.altair_chart(chart_ttr, use_container_width=True)
                
            # --- Types vs Tokens scatter with optional Color by Group ---
            scatter_ctrl_col, scatter_chart_col = st.columns([1, 3])
            with scatter_ctrl_col:
                with st.form(key="color_form"):
                    color_by_group = st.checkbox("Color by group", value=True, key="color_by_group")
                    group_colors = {}
                    if color_by_group and 'Level' in df.columns:
                        groups = sorted(df['Level'].dropna().unique().tolist())
                        default_palette = [
                            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                            "#bcbd22", "#17becf"
                        ]
                        for i, grp in enumerate(groups):
                            default_hex = default_palette[i % len(default_palette)]
                            c1, c2 = st.columns([1, 6], gap="small")
                            with c1:
                                picked = st.color_picker(grp, value=default_hex, key=f"color_{grp}", label_visibility="collapsed")
                            with c2:
                                st.markdown(f"<p style='font-size: 0.85rem; margin-top: 5px;'>{grp}</p>", unsafe_allow_html=True)
                            group_colors[grp] = picked
                    st.form_submit_button("Apply")

            with scatter_chart_col:
                if color_by_group and 'Level' in df.columns and group_colors:
                    scatter_df = df.copy()
                    scatter_df['_color'] = scatter_df['Level'].map(group_colors).fillna('#888888')
                    # Build per-group layers so custom colors apply
                    layers = []
                    for grp, hex_color in group_colors.items():
                        grp_df = scatter_df[scatter_df['Level'] == grp]
                        if grp_df.empty:
                            continue
                        layer = alt.Chart(grp_df).mark_circle(size=60, color=hex_color).encode(
                            x=alt.X('Tokens_total:Q', title='Total Tokens (Text Length)'),
                            y=alt.Y('Types_total:Q', title='Total Types (Unique Words)'),
                            tooltip=['ID', 'Level', 'Tokens_total', 'Types_total']
                        ).properties(title='Tokens vs. Types')
                        layers.append(layer)
                    if layers:
                        chart_scatter = alt.layer(*layers)
                        st.altair_chart(chart_scatter, use_container_width=True)
                else:
                    chart_scatter = alt.Chart(df).mark_circle(size=60).encode(
                        x=alt.X('Tokens_total:Q', title='Total Tokens (Text Length)'),
                        y=alt.Y('Types_total:Q', title='Total Types (Unique Words)'),
                        tooltip=['ID', 'Tokens_total', 'Types_total']
                    ).properties(title='Tokens vs. Types')
                    st.altair_chart(chart_scatter, use_container_width=True)

            st.divider()

            st.subheader("Part-of-Speech (POS) Distribution")
            # Calculate aggregate POS percentages
            total_cw = df['CW'].sum()
            if total_cw > 0:
                pos_sums = {
                    "Noun": df['NOUN_tokens'].sum(),
                    "Verb": df['VERB_tokens'].sum(),
                    "Adjective": df['ADJ_tokens'].sum(),
                    "Adverb": df['ADV_tokens'].sum(),
                }
                pos_df = pd.DataFrame(list(pos_sums.items()), columns=['POS', 'Count'])
                
                max_pos_count = pos_df['Count'].max()
                pos_domain_max = max_pos_count * 1.2 if pd.notnull(max_pos_count) and max_pos_count > 0 else 10
                
                chart_pos_base = alt.Chart(pos_df).encode(
                    y=alt.Y('POS:N', sort=["Noun", "Verb", "Adjective", "Adverb"], title=None),
                    x=alt.X('Count:Q', title='Total Tokens', scale=alt.Scale(domain=[0, pos_domain_max])),
                    color=alt.Color('POS:N', 
                        scale=alt.Scale(
                            domain=["Noun", "Verb", "Adjective", "Adverb"],
                            range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                        ), 
                        legend=None
                    ),
                    tooltip=["POS", "Count"]
                )
                bars_pos = chart_pos_base.mark_bar()
                text_pos = chart_pos_base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3
                ).encode(
                    text='Count:Q'
                )
                chart_pos = (bars_pos + text_pos).properties(title='Proportion of Lexical Categories')
                st.altair_chart(chart_pos, use_container_width=True)
                
                st.markdown("### Top Lexical Items by POS")
                if 'Text' in df.columns:
                    c1, c2 = st.columns(2)
                    top_pos_n = c1.number_input("Top N items", min_value=1, max_value=50, value=10, key="top_pos_n")
                    pos_format = c2.radio("Count by", ["Lemma", "Token"], index=0, horizontal=True)
                    use_lemma_flag = (pos_format == "Lemma")
                    
                    texts_tuple = tuple(df['Text'].dropna().astype(str))
                    with st.spinner("Extracting POS items..."):
                        nouns_c, verbs_c, adjs_c, advs_c = get_pos_distribution(texts_tuple, use_lemma_flag)
                    
                    pos_displays = [
                        ("Noun", nouns_c, '#1f77b4'),
                        ("Verb", verbs_c, '#ff7f0e'),
                        ("Adjective", adjs_c, '#2ca02c'),
                        ("Adverb", advs_c, '#d62728')
                    ]
                    
                    cols = st.columns(4)
                    for idx, (p_name, p_counter, p_color) in enumerate(pos_displays):
                        with cols[idx]:
                            top_items = [
                                (item, freq) for item, freq in p_counter.most_common(int(top_pos_n))
                            ]
                            if top_items:
                                sub_df = pd.DataFrame(top_items, columns=['Item', 'Frequency'])
                                
                                max_freq = sub_df['Frequency'].max()
                                domain_max = max_freq * 1.3 if pd.notnull(max_freq) and max_freq > 0 else 10
                                
                                sub_chart = alt.Chart(sub_df).mark_bar(color=p_color).encode(
                                    x=alt.X('Frequency:Q', title=None, axis=alt.Axis(tickMinStep=1, format='d'), scale=alt.Scale(domain=[0, domain_max])),
                                    y=alt.Y('Item:N', sort='-x', title=None),
                                    tooltip=['Item', 'Frequency']
                                ).properties(
                                    title=f"Top {p_name}s",
                                    height=max(200, 30 * len(top_items))
                                )
                                sub_text = sub_chart.mark_text(
                                    align='left',
                                    baseline='middle',
                                    dx=3
                                ).encode(
                                    text='Frequency:Q'
                                )
                                st.altair_chart(sub_chart + sub_text, use_container_width=True)
                            else:
                                st.info(f"No {p_name}s found")
                else:
                    st.warning("Original text data is not available, so top lexical items cannot be extracted.")
            else:
                st.info("Not enough content words to analyze POS distribution.")

        elif st.session_state["active_view"] == "Corpus Tools":

            # Helper: collect all texts from the filtered results
            def get_all_texts():
                """Return list of (doc_id, text) from current results."""
                pairs = []
                for r in results:
                    text = r.get("Text", "")
                    if text:
                        pairs.append((r.get("ID", "?"), text))
                return pairs

            def tokenize_simple(text):
                """Basic whitespace-aware word tokenizer (lowercase, alpha only)."""
                return re.findall(r"[a-zA-Z']+", text.lower())

            # ── 1. Keyword Search & KWIC Concordance ─────────────────────────
            st.subheader("Keyword Search & Concordance (KWIC)")
            kwic_query = st.text_input(
                "Search keyword or phrase",
                placeholder="e.g.  university",
                key="kwic_query"
            )
            context_window = st.slider("Context window (words each side)", 3, 30, 10, key="kwic_window")

            if kwic_query.strip():
                query_lower = kwic_query.strip().lower()
                kwic_rows = []
                all_texts = get_all_texts()
                for doc_id, text in all_texts:
                    words = re.findall(r"[\w']+", text)  # preserve case for display
                    lower_words = [w.lower() for w in words]
                    # Support multi-word phrases
                    q_tokens = query_lower.split()
                    q_len = len(q_tokens)
                    for i, _ in enumerate(lower_words):
                        if lower_words[i:i + q_len] == q_tokens:
                            left = " ".join(words[max(0, i - context_window):i])
                            node = " ".join(words[i:i + q_len])
                            right = " ".join(words[i + q_len:i + q_len + context_window])
                            kwic_rows.append({
                                "Document": doc_id,
                                "Left context": left,
                                "Node": node,
                                "Right context": right,
                            })

                if kwic_rows:
                    st.markdown(f"**{len(kwic_rows)} concordance line(s)** for *{kwic_query}*")
                    kwic_df = pd.DataFrame(kwic_rows)
                    st.dataframe(
                        kwic_df.style.applymap(
                            lambda v: "font-weight: bold; color: #c0392b;",
                            subset=["Node"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    # Download
                    kwic_csv = io.StringIO()
                    kwic_df.to_csv(kwic_csv, index=False)
                    st.download_button(
                        "Download KWIC as CSV",
                        data=kwic_csv.getvalue(),
                        file_name="kwic_results.csv",
                        mime="text/csv",
                        key="kwic_download"
                    )
                else:
                    st.info(f'No matches found for "{kwic_query}" in the current corpus.')

            st.divider()

            # ── 2. N-gram Analysis ────────────────────────────────────────────
            st.subheader("N-gram Analysis")
            ngram_col1, ngram_col2 = st.columns([1, 3])
            
            with ngram_col1:
                with st.form("ngram_form"):
                    ngram_n = st.number_input("N-gram size", min_value=1, max_value=10, value=2, step=1, key="ngram_n")
                    ngram_topk = st.number_input("Top N results", min_value=5, max_value=200, value=20, step=5, key="ngram_topk")
                    ngram_minfreq = st.number_input("Min frequency", min_value=1, value=2, step=1, key="ngram_minfreq")
                    
                    exclude_func = st.checkbox("Exclude function words", value=False)
                    exclude_stop = st.checkbox("Exclude stop words", value=False)
                    
                    # Pre-calculate default stop words if the dataset matches
                    initial_stops = ""
                    if st.session_state.get("current_dataset") == "openalex_5k":
                        initial_stops = "artificial intelligence, ai, criticality, critical thinking"
                    
                    stop_words_str = st.text_area("Stop words list", value=initial_stops, help="Enter words or phrases separated by commas.")
                    apply_btn = st.form_submit_button("Apply")

            with ngram_col2:
                all_texts = get_all_texts()
                ngram_counter: Counter = Counter()
                
                # Parse stop words
                stop_list = [s.strip().lower() for s in stop_words_str.split(',') if s.strip()]
                
                # Standard function words list
                func_words = {
                    'the', 'a', 'an', 'and', 'but', 'or', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 
                    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
                    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
                    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'shall', 'should', 'can', 'could', 'may', 'might', 
                    'must', 'ought'
                }
                
                for _, text in all_texts:
                    tokens = tokenize_simple(text)
                    grams = list(zip(*[tokens[i:] for i in range(ngram_n)]))
                    for g in grams:
                        g_str = " ".join(g)
                        
                        # Exclude function words
                        if exclude_func:
                            if any(word in func_words for word in g):
                                continue
                        
                        # Exclude user-defined stop words
                        if exclude_stop:
                            # 1. If any component word of the g-gram is in the stop list as a unigram
                            # 2. Or if the stop list item (which might be a phrase) is a sub-part of g_str
                            found_stop = False
                            for stop_item in stop_list:
                                if f" {stop_item} " in f" {g_str} ":
                                    found_stop = True
                                    break
                                if any(word == stop_item for word in g):
                                    found_stop = True
                                    break
                            if found_stop:
                                continue
                                
                        ngram_counter[g] += 1

                top_ngrams = [
                    (" ".join(gram), freq)
                    for gram, freq in ngram_counter.most_common()
                    if freq >= ngram_minfreq
                ][:int(ngram_topk)]

                if top_ngrams:
                    ngram_df = pd.DataFrame(top_ngrams, columns=["N-gram", "Frequency"])
                    base_ngram = alt.Chart(ngram_df).encode(
                        x=alt.X("Frequency:Q", axis=alt.Axis(tickMinStep=1, format='d')),
                        y=alt.Y("N-gram:N", sort="-x", axis=alt.Axis(labelLimit=0)),
                        tooltip=["N-gram", "Frequency"]
                    )
                    bars_ngram = base_ngram.mark_bar()
                    text_ngram = base_ngram.mark_text(align='left', baseline='middle', dx=3).encode(text='Frequency:Q')
                    chart_ngram = (bars_ngram + text_ngram).properties(title=f"Top {len(top_ngrams)} {ngram_n}-grams")
                    st.altair_chart(chart_ngram, use_container_width=True)
                    st.caption("Tip: Hover over the top right corner of the chart and click the expand icon to view it in full screen.")
                    ngram_csv = io.StringIO()
                    ngram_df.to_csv(ngram_csv, index=False)
                    st.download_button(
                        "Download N-grams as CSV",
                        data=ngram_csv.getvalue(),
                        file_name="ngrams.csv",
                        mime="text/csv",
                        key="ngram_download"
                    )
                else:
                    st.info("No n-grams meet the minimum frequency threshold.")

            st.divider()

            # ── 3. Collocation ────────────────────────────────────────────────
            st.subheader("Collocation")
            coll_query = st.text_input(
                "Node word for collocation",
                placeholder="e.g.  student",
                key="coll_query"
            )
            coll_col1, coll_col2 = st.columns([1, 3])
            with coll_col1:
                slider_col1, slider_col2 = st.columns(2)
                with slider_col1:
                    coll_window_left = st.slider("Left window", 1, 10, 4, key="coll_window_left")
                with slider_col2:
                    coll_window_right = st.slider("Right window", 1, 10, 4, key="coll_window_right")
                coll_topk = st.number_input("Top collocates", min_value=5, max_value=100, value=20, step=5, key="coll_topk")
                coll_minfreq = st.number_input("Min co-occurrence", min_value=1, value=2, step=1, key="coll_minfreq")

            with coll_col2:
                if coll_query.strip():
                    node_lower = coll_query.strip().lower()
                    left_counter: Counter = Counter()
                    right_counter: Counter = Counter()
                    node_freq = 0
                    all_texts = get_all_texts()
                    for _, text in all_texts:
                        tokens = tokenize_simple(text)
                        for idx, tok in enumerate(tokens):
                            if tok == node_lower:
                                node_freq += 1
                                # Collect left context
                                left_start = max(0, idx - coll_window_left)
                                left_toks = tokens[left_start:idx]
                                left_counter.update(left_toks)
                                # Collect right context
                                right_end = min(len(tokens), idx + coll_window_right + 1)
                                right_toks = tokens[idx + 1:right_end]
                                right_counter.update(right_toks)

                    # Remove the node word itself from collocates
                    left_counter.pop(node_lower, None)
                    right_counter.pop(node_lower, None)

                    top_left = [
                        (word, freq)
                        for word, freq in left_counter.most_common()
                        if freq >= coll_minfreq
                    ][:int(coll_topk)]

                    top_right = [
                        (word, freq)
                        for word, freq in right_counter.most_common()
                        if freq >= coll_minfreq
                    ][:int(coll_topk)]

                    if node_freq == 0:
                        st.info(f'Node word "{coll_query}" not found in corpus.')
                    elif top_left or top_right:
                        st.markdown(f"Node **{coll_query}** appears **{node_freq}** time(s).")

                        # Display left and right collocates side by side
                        left_display_col, right_display_col = st.columns(2)

                        with left_display_col:
                            if top_left:
                                st.markdown(f"**Left collocates** (−{coll_window_left} words)")
                                left_df = pd.DataFrame(top_left, columns=["Collocate", "Co-occurrence"])
                                base_left = alt.Chart(left_df).encode(
                                    x=alt.X("Co-occurrence:Q", axis=alt.Axis(tickMinStep=1, format='d')),
                                    y=alt.Y("Collocate:N", sort="-x"),
                                    tooltip=["Collocate", "Co-occurrence"]
                                )
                                bars_left = base_left.mark_bar(color="#3498db")
                                text_left = base_left.mark_text(align='left', baseline='middle', dx=3).encode(text='Co-occurrence:Q')
                                chart_left = (bars_left + text_left).properties(title=f"Left collocates", height=300)
                                st.altair_chart(chart_left, use_container_width=True)
                                st.caption("Tip: Hover over the top right to expand.")
                                left_csv = io.StringIO()
                                left_df.to_csv(left_csv, index=False)
                                st.download_button(
                                    "Download Left as CSV",
                                    data=left_csv.getvalue(),
                                    file_name="left_collocates.csv",
                                    mime="text/csv",
                                    key="left_coll_download"
                                )
                            else:
                                st.info("No left collocates meet the minimum frequency threshold.")

                        with right_display_col:
                            if top_right:
                                st.markdown(f"**Right collocates** (+{coll_window_right} words)")
                                right_df = pd.DataFrame(top_right, columns=["Collocate", "Co-occurrence"])
                                base_right = alt.Chart(right_df).encode(
                                    x=alt.X("Co-occurrence:Q", axis=alt.Axis(tickMinStep=1, format='d')),
                                    y=alt.Y("Collocate:N", sort="-x"),
                                    tooltip=["Collocate", "Co-occurrence"]
                                )
                                bars_right = base_right.mark_bar(color="#e74c3c")
                                text_right = base_right.mark_text(align='left', baseline='middle', dx=3).encode(text='Co-occurrence:Q')
                                chart_right = (bars_right + text_right).properties(title=f"Right collocates", height=300)
                                st.altair_chart(chart_right, use_container_width=True)
                                st.caption("Tip: Hover over the top right to expand.")
                                right_csv = io.StringIO()
                                right_df.to_csv(right_csv, index=False)
                                st.download_button(
                                    "Download Right as CSV",
                                    data=right_csv.getvalue(),
                                    file_name="right_collocates.csv",
                                    mime="text/csv",
                                    key="right_coll_download"
                                )
                            else:
                                st.info("No right collocates meet the minimum frequency threshold.")
                    else:
                        st.info("No collocates meet the minimum frequency threshold.")
                else:
                    st.info("Enter a node word above to compute collocations.")
