import streamlit as st
import pandas as pd
import io
from typing import Set, Tuple

def load_dictionary_from_text(dict_text: str) -> Tuple[Set[str], Set[str]]:
    """Load positive/negative words from dictionary text"""
    lines = [line.strip().lower() for line in dict_text.split('\n') if line.strip()]
    
    positive_words = set()
    negative_words = set()
    
    for line in lines:
        if ',' in line:
            word, sentiment = line.split(',', 1)
            word = word.strip()
            sentiment = sentiment.strip()
            
            if sentiment in ['positive', 'pos', '1']:
                positive_words.add(word)
            elif sentiment in ['negative', 'neg', '0', '-1']:
                negative_words.add(word)
        else:
            # If no sentiment specified, treat as positive by default
            positive_words.add(line)
    
    return positive_words, negative_words

def load_dictionary_from_file(uploaded_file) -> Tuple[Set[str], Set[str]]:
    """Load positive/negative words from uploaded dictionary file"""
    content = uploaded_file.read().decode('utf-8')
    return load_dictionary_from_text(content)

def continuous_sentiment_score(text, positive_words: Set[str], negative_words: Set[str]) -> float:
    """Calculate continuous sentiment score (-1 to +1)"""
    if not text or pd.isna(text):
        return 0.0
    
    words = str(text).lower().split()
    if not words:
        return 0.0
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Normalized difference: ranges from -1 to +1
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words == 0:
        return 0.0
    
    return (pos_count - neg_count) / total_sentiment_words

def process_sentiment_data(df: pd.DataFrame, text_column: str, positive_words: Set[str], negative_words: Set[str]) -> pd.DataFrame:
    """Main processing function"""
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate continuous sentiment scores
    result_df['sentiment_score'] = result_df[text_column].apply(
        lambda x: continuous_sentiment_score(x, positive_words, negative_words)
    )
    
    # Add sentiment labels for easier interpretation
    result_df['sentiment_label'] = result_df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
    )
    
    return result_df

# Default dictionary content
DEFAULT_DICTIONARY = """good,positive
great,positive
excellent,positive
amazing,positive
wonderful,positive
fantastic,positive
awesome,positive
love,positive
like,positive
happy,positive
joy,positive
pleased,positive
satisfied,positive
delighted,positive
thrilled,positive
bad,negative
terrible,negative
awful,negative
horrible,negative
hate,negative
dislike,negative
sad,negative
angry,negative
frustrated,negative
disappointed,negative
upset,negative
annoyed,negative
disgusted,negative
furious,negative
miserable,negative"""

def main():
    st.set_page_config(
        page_title="Sentiment Analysis Tool",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Sentiment Analysis Tool")
    st.markdown("Upload your dataset and customize the sentiment dictionary to analyze text sentiment.")
    
    # Sidebar for dictionary management
    st.sidebar.header("📚 Sentiment Dictionary")
    
    dict_option = st.sidebar.radio(
        "Choose dictionary source:",
        ["Use default dictionary", "Upload dictionary file", "Edit dictionary manually"]
    )
    
    positive_words = set()
    negative_words = set()
    
    if dict_option == "Use default dictionary":
        positive_words, negative_words = load_dictionary_from_text(DEFAULT_DICTIONARY)
        st.sidebar.success(f"Using default dictionary: {len(positive_words)} positive, {len(negative_words)} negative words")
        
        with st.sidebar.expander("View default dictionary"):
            st.text(DEFAULT_DICTIONARY)
    
    elif dict_option == "Upload dictionary file":
        uploaded_dict = st.sidebar.file_uploader(
            "Upload dictionary file (CSV/TXT format: word,sentiment)",
            type=['csv', 'txt'],
            help="Format: word,positive or word,negative (one per line)"
        )
        
        if uploaded_dict:
            positive_words, negative_words = load_dictionary_from_file(uploaded_dict)
            st.sidebar.success(f"Loaded dictionary: {len(positive_words)} positive, {len(negative_words)} negative words")
    
    elif dict_option == "Edit dictionary manually":
        st.sidebar.markdown("**Edit the dictionary below:**")
        dict_text = st.sidebar.text_area(
            "Dictionary (word,sentiment format):",
            value=DEFAULT_DICTIONARY,
            height=300,
            help="Format: word,positive or word,negative (one per line)"
        )
        
        if dict_text:
            positive_words, negative_words = load_dictionary_from_text(dict_text)
            st.sidebar.success(f"Dictionary loaded: {len(positive_words)} positive, {len(negative_words)} negative words")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file containing text data for sentiment analysis"
        )
        
        if uploaded_file and (positive_words or negative_words):
            try:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                st.success(f"Dataset loaded successfully! Shape: {df.shape}")
                
                # Show data preview
                with st.expander("Preview dataset"):
                    st.dataframe(df.head())
                
                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the text column for sentiment analysis:",
                        text_columns,
                        help="Choose the column containing the text to analyze"
                    )
                    
                    if st.button("🚀 Analyze Sentiment", type="primary"):
                        with st.spinner("Analyzing sentiment..."):
                            # Process sentiment analysis
                            result_df = process_sentiment_data(df, selected_column, positive_words, negative_words)
                            
                            st.success("Sentiment analysis completed!")
                            
                            # Display results
                            st.header("📈 Results")
                            
                            # Summary statistics
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            with col_a:
                                positive_count = (result_df['sentiment_label'] == 'Positive').sum()
                                st.metric("Positive", positive_count)
                            
                            with col_b:
                                neutral_count = (result_df['sentiment_label'] == 'Neutral').sum()
                                st.metric("Neutral", neutral_count)
                            
                            with col_c:
                                negative_count = (result_df['sentiment_label'] == 'Negative').sum()
                                st.metric("Negative", negative_count)
                            
                            with col_d:
                                avg_score = result_df['sentiment_score'].mean()
                                st.metric("Avg Score", f"{avg_score:.3f}")
                            
                            # Results table
                            st.subheader("Detailed Results")
                            
                            # Filter options
                            filter_option = st.selectbox(
                                "Filter results:",
                                ["All", "Positive", "Neutral", "Negative"]
                            )
                            
                            if filter_option != "All":
                                filtered_df = result_df[result_df['sentiment_label'] == filter_option]
                            else:
                                filtered_df = result_df
                            
                            # Display results with color coding
                            def color_sentiment(val):
                                if val > 0.1:
                                    return 'background-color: #d4edda'  # light green
                                elif val < -0.1:
                                    return 'background-color: #f8d7da'  # light red
                                else:
                                    return 'background-color: #fff3cd'  # light yellow
                            
                            styled_df = filtered_df.style.applymap(
                                color_sentiment, 
                                subset=['sentiment_score']
                            ).format({'sentiment_score': '{:.3f}'})
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Download results
                            csv_buffer = io.StringIO()
                            result_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_data,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
                            
                            # Sentiment distribution chart
                            st.subheader("📊 Sentiment Distribution")
                            sentiment_counts = result_df['sentiment_label'].value_counts()
                            st.bar_chart(sentiment_counts)
                
                else:
                    st.error("No text columns found in the dataset. Please upload a CSV with text data.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. **Choose Dictionary Source**: 
           - Use default dictionary
           - Upload your own dictionary file
           - Edit dictionary manually
        
        2. **Upload Dataset**: 
           - Upload a CSV file with text data
           - Select the text column to analyze
        
        3. **Analyze**: 
           - Click "Analyze Sentiment" button
           - View results and download CSV
        
        **Dictionary Format:**
        ```
        word,positive
        word,negative
        ```
        
        **Sentiment Score Range:**
        - **+1.0**: Most positive
        - **0.0**: Neutral
        - **-1.0**: Most negative
        """)
        
        if positive_words or negative_words:
            st.header("📖 Current Dictionary")
            st.markdown(f"**Positive words:** {len(positive_words)}")
            st.markdown(f"**Negative words:** {len(negative_words)}")
            
            if len(positive_words) > 0:
                with st.expander("View positive words"):
                    st.write(", ".join(sorted(list(positive_words)[:20])))
                    if len(positive_words) > 20:
                        st.write(f"... and {len(positive_words) - 20} more")
            
            if len(negative_words) > 0:
                with st.expander("View negative words"):
                    st.write(", ".join(sorted(list(negative_words)[:20])))
                    if len(negative_words) > 20:
                        st.write(f"... and {len(negative_words) - 20} more")

if __name__ == "__main__":
    main()
