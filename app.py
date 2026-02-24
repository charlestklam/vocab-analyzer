import streamlit as st
import pandas as pd
import json
import io
import altair as alt
from analyzer import LexicalAnalyzer

st.set_page_config(page_title="Vocabulary Analyzer", layout="wide")

st.title("Vocabulary Development Analysis App")
st.markdown("""
Upload individual `.txt` files or a combined `.json` file to calculate syntactic and lexical frequency metrics.
Paragraphs will be analyzed structurally using `spaCy` NLP and mapped against a synthesised English 25k frequency list.
""")

st.sidebar.header("Options")
file_type = st.sidebar.radio("Upload Format", ["Text Files (.txt)", "JSON File (.json)"], help="Choose whether you are uploading multiple TXT files or a structured JSON dump.")

uploaded_files = st.sidebar.file_uploader(
    "Choose files", 
    type=["txt"] if "Text" in file_type else ["json"], 
    accept_multiple_files=True if "Text" in file_type else False
)

@st.cache_resource
def load_analyzer():
    # Cache the NLP model and dictionary generation to prevent reloading per run
    return LexicalAnalyzer()

analyzer = load_analyzer()

if "results" not in st.session_state:
    st.session_state["results"] = []

st.sidebar.header("Presets")
if st.sidebar.button("Grade 7 Writing (n=50)"):
    with st.spinner("Loading Grade 7 Writing..."):
        try:
            with open('data/batch_50_results.json', 'r', encoding='utf-8') as f:
                st.session_state["results"] = json.load(f)
        except Exception as e:
            st.error(f"Error loading preset: {e}")

if st.sidebar.button("AI Criticality Abstracts (n=5525)"):
    with st.spinner("Loading AI Criticality Abstracts..."):
        try:
            with open('data/openalex_5525_results.json', 'r', encoding='utf-8') as f:
                st.session_state["results"] = json.load(f)
        except Exception as e:
            st.error(f"Error loading preset: {e}")

st.sidebar.header("Filters")
col1, col2 = st.sidebar.columns(2)
min_words = col1.number_input("Min Words", min_value=0, value=0, step=10)
max_words = col2.number_input("Max Words", min_value=0, value=10000, step=10)

available_groups = []
if st.session_state["results"]:
    available_groups = sorted(list(set(str(r.get("Level", "Unknown")) for r in st.session_state["results"])))
selected_groups = st.sidebar.multiselect("Filter by Group(s)", options=available_groups, default=available_groups)

if st.button("Run Analysis on Uploaded Data"):
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
        
        tab1, tab2 = st.tabs(["Data Table", "Visualization"])
        
        with tab1:
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
            
        with tab2:
            st.subheader("Lexical Frequency Profile (K-Bands)")
            
            # Aggregate K-bands into tiers
            tier_data = {
                "Text ID": [],
                "Level": [],
                "Tier": [],
                "Count": []
            }
            
            for _, row in df.iterrows():
                doc_id = row['ID']
                level = row['Level']
                
                k1 = row['K1']
                k2_k5 = sum(row[f'K{i}'] for i in range(2, 6))
                k6_k10 = sum(row[f'K{i}'] for i in range(6, 11))
                k11_k25 = sum(row[f'K{i}'] for i in range(11, 26))
                off = row['OFF']
                
                total_words = k1 + k2_k5 + k6_k10 + k11_k25 + off
                if total_words > 0:
                    tier_data["Text ID"].extend([doc_id] * 5)
                    tier_data["Level"].extend([level] * 5)
                    tier_data["Tier"].extend(["K1", "K2-K5", "K6-K10", "K11-K25", "OFF"])
                    tier_data["Count"].extend([
                        k1 / total_words, 
                        k2_k5 / total_words, 
                        k6_k10 / total_words, 
                        k11_k25 / total_words, 
                        off / total_words
                    ])
            
            if tier_data["Tier"]:
                tier_df = pd.DataFrame(tier_data)
                chart_lfp = alt.Chart(tier_df).mark_bar().encode(
                    x=alt.X('Tier:O', sort=["K1", "K2-K5", "K6-K10", "K11-K25", "OFF"]),
                    y=alt.Y('mean(Count):Q', axis=alt.Axis(format='%', title='Percentage of Text')),
                    tooltip=['Tier', alt.Tooltip('mean(Count):Q', format='.2%')]
                ).properties(
                    title='Average Lexical Frequency Profile Across Corpus'
                )
                st.altair_chart(chart_lfp, use_container_width=True)

            st.divider()

            st.subheader("Syntactic Complexity & Lexical Richness")
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot of Lexical Density
                chart_ld = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    y=alt.Y('LD:Q', title='Lexical Density (Content Words / Total)'),
                    x=alt.X('Level:N', title='Group') if 'Level' in df.columns and df['Level'].nunique() > 1 else alt.value('Corpus')
                ).properties(title='Lexical Density Distribution')
                st.altair_chart(chart_ld, use_container_width=True)
                
            with col2:
                # Boxplot of TTR
                chart_ttr = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    y=alt.Y('TTR:Q', title='Type-Token Ratio'),
                    x=alt.X('Level:N', title='Group') if 'Level' in df.columns and df['Level'].nunique() > 1 else alt.value('Corpus')
                ).properties(title='TTR Distribution')
                st.altair_chart(chart_ttr, use_container_width=True)
                
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
                    "Nouns": df['NOUN_tokens'].sum(),
                    "Verbs": df['VERB_tokens'].sum(),
                    "Adjectives": df['ADJ_tokens'].sum(),
                    "Adverbs": df['ADV_tokens'].sum(),
                }
                pos_df = pd.DataFrame(list(pos_sums.items()), columns=['POS', 'Count'])
                
                base = alt.Chart(pos_df).encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    color=alt.Color("POS:N"),
                    tooltip=["POS", "Count"]
                )
                pie = base.mark_arc(outerRadius=120)
                text = base.mark_text(radius=150, size=12).encode(
                    text="POS:N"
                )
                
                chart_pos = (pie + text).properties(title='Proportion of Lexical Categories')
                st.altair_chart(chart_pos, use_container_width=True)
            else:
                st.info("Not enough content words to analyze POS distribution.")
