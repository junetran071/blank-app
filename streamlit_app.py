import streamlit as st
import pandas as pd
import re
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Text Classification App",
    page_icon="📊",
    layout="wide"
)

# Default dictionaries
DEFAULT_DICTIONARIES = {
    'urgency_marketing': {
        'limited', 'limited time', 'limited run', 'limited edition', 'order now',
        'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
        'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
        'expires soon', 'final hours', 'almost gone'
    },
    'exclusive_marketing': {
        'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
        'members only', 'vip', 'special access', 'invitation only',
        'premium', 'privileged', 'limited access', 'select customers',
        'insider', 'private sale', 'early access'
    }
}

def classify_text(text, dictionaries):
    """Classify text based on dictionary matches"""
    if pd.isna(text):
        return {}
    
    text_lower = text.lower()
    results = {}
    
    for dict_name, terms in dictionaries.items():
        matches = []
        for term in terms:
            if term.lower() in text_lower:
                matches.append(term)
        
        results[dict_name] = {
            'count': len(matches),
            'matches': matches,
            'present': len(matches) > 0
        }
    
    return results

def process_dataframe(df, text_column, dictionaries):
    """Process DataFrame and add classification results"""
    # Apply classification
    classifications = df[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Create a copy of the dataframe to avoid modifying the original
    df_results = df.copy()
    
    # Add results as new columns
    for dict_name in dictionaries.keys():
        df_results[f'{dict_name}_count'] = classifications.apply(
            lambda x: x.get(dict_name, {}).get('count', 0)
        )
        df_results[f'{dict_name}_present'] = classifications.apply(
            lambda x: x.get(dict_name, {}).get('present', False)
        )
        df_results[f'{dict_name}_matches'] = classifications.apply(
            lambda x: ', '.join(x.get(dict_name, {}).get('matches', []))
        )
    
    return df_results

def main():
    st.title("📊 Text Classification App")
    st.write("Upload your dataset and classify text using customizable dictionaries")
    
    # Sidebar for dictionary management
    st.sidebar.header("📝 Dictionary Management")
    
    # Initialize session state for dictionaries
    if 'dictionaries' not in st.session_state:
        st.session_state.dictionaries = DEFAULT_DICTIONARIES.copy()
    
    # Dictionary editor
    st.sidebar.subheader("Edit Dictionaries")
    
    # Select dictionary to edit
    dict_names = list(st.session_state.dictionaries.keys())
    selected_dict = st.sidebar.selectbox("Select Dictionary:", dict_names)
    
    if selected_dict:
        st.sidebar.write(f"**{selected_dict}** terms:")
        
        # Display current terms
        current_terms = list(st.session_state.dictionaries[selected_dict])
        terms_text = '\n'.join(current_terms)
        
        # Text area for editing terms
        edited_terms = st.sidebar.text_area(
            "Edit terms (one per line):",
            value=terms_text,
            height=200,
            key=f"terms_{selected_dict}"
        )
        
        # Update dictionary
        if st.sidebar.button("Update Dictionary", key=f"update_{selected_dict}"):
            new_terms = set([term.strip() for term in edited_terms.split('\n') if term.strip()])
            st.session_state.dictionaries[selected_dict] = new_terms
            st.sidebar.success(f"Updated {selected_dict}!")
    
    # Add new dictionary
    st.sidebar.subheader("Add New Dictionary")
    new_dict_name = st.sidebar.text_input("Dictionary Name:")
    new_dict_terms = st.sidebar.text_area("Terms (one per line):", height=100)
    
    if st.sidebar.button("Add Dictionary") and new_dict_name and new_dict_terms:
        terms_set = set([term.strip() for term in new_dict_terms.split('\n') if term.strip()])
        st.session_state.dictionaries[new_dict_name] = terms_set
        st.sidebar.success(f"Added {new_dict_name}!")
        st.rerun()
    
    # Reset to defaults
    if st.sidebar.button("Reset to Defaults"):
        st.session_state.dictionaries = DEFAULT_DICTIONARIES.copy()
        st.sidebar.success("Reset to default dictionaries!")
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Dataset")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file containing text data to classify"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
                
                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the text column to classify:",
                        text_columns,
                        help="Choose the column containing the text you want to classify"
                    )
                    
                    # Show sample data
                    st.subheader("📋 Sample Data")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Process button
                    if st.button("🚀 Classify Text", type="primary"):
                        with st.spinner("Processing..."):
                            # Process the dataframe
                            df_results = process_dataframe(
                                df, selected_column, st.session_state.dictionaries
                            )
                            
                            # Store results in session state
                            st.session_state.results = df_results
                            
                            st.success("Classification completed!")
                    
                    # Display results if available
                    if 'results' in st.session_state:
                        st.subheader("📊 Classification Results")
                        
                        # Summary statistics
                        st.write("**Summary Statistics:**")
                        summary_cols = st.columns(len(st.session_state.dictionaries))
                        
                        for i, dict_name in enumerate(st.session_state.dictionaries.keys()):
                            with summary_cols[i]:
                                present_count = st.session_state.results[f'{dict_name}_present'].sum()
                                total_count = len(st.session_state.results)
                                percentage = (present_count / total_count) * 100
                                
                                st.metric(
                                    label=f"{dict_name.replace('_', ' ').title()}",
                                    value=f"{present_count}/{total_count}",
                                    delta=f"{percentage:.1f}%"
                                )
                        
                        # Show results table
                        st.write("**Detailed Results:**")
                        
                        # Select columns to display
                        result_columns = [selected_column]
                        for dict_name in st.session_state.dictionaries.keys():
                            result_columns.extend([
                                f'{dict_name}_count',
                                f'{dict_name}_present',
                                f'{dict_name}_matches'
                            ])
                        
                        # Filter columns that exist in the dataframe
                        available_columns = [col for col in result_columns if col in st.session_state.results.columns]
                        
                        st.dataframe(
                            st.session_state.results[available_columns],
                            use_container_width=True
                        )
                        
                        # Download button
                        csv_data = st.session_state.results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name="classified_data.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("No text columns found in the uploaded file.")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        else:
            # Show sample data format
            st.info("👆 Upload a CSV file to get started")
            st.write("**Expected CSV format:**")
            sample_data = pd.DataFrame({
                'ID': [1, 2, 3],
                'Statement': [
                    'Limited time offer - act now!',
                    'Exclusive deal for VIP members only',
                    'Regular product description'
                ]
            })
            st.dataframe(sample_data, use_container_width=True)
    
    with col2:
        st.header("📚 Current Dictionaries")
        
        for dict_name, terms in st.session_state.dictionaries.items():
            with st.expander(f"{dict_name.replace('_', ' ').title()} ({len(terms)} terms)"):
                st.write(", ".join(sorted(terms)))
        
        st.subheader("ℹ️ How it works")
        st.write("""
        1. **Upload** your CSV file
        2. **Select** the text column to analyze
        3. **Customize** dictionaries in the sidebar
        4. **Click** 'Classify Text' to process
        5. **Download** the results with new classification columns
        
        The app will add these columns for each dictionary:
        - `*_count`: Number of matching terms
        - `*_present`: Boolean indicating presence
        - `*_matches`: List of matched terms
        """)

if __name__ == "__main__":
    main()
