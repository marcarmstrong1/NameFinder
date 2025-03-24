import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

st.title("A helpful tool to find names within reviews")

rating = st.slider("Select a rating (Ex. 4 would mean 4 stars and up)", 1, 5)

data = st.file_uploader("Upload a XLSX file", type=["xlsx"])

if data is not None:
    df = pd.read_excel(data)
    df.dropna(subset=["ACTUAL REVIEW"], inplace=True)

    df = df[['DATE','PROPERTY NAME','ACTUAL REVIEW', "STAR RATING", "REVIEWID"]]
    df = df[df["STAR RATING"] >= int(rating)]
    
    @st.cache(allow_output_mutation=True)
    def load_model():
        bert_tokenizer = AutoTokenizer.from_pretrained('dslim/bert-large-NER')
        bert_model = AutoModelForTokenClassification.from_pretrained('dslim/bert-large-NER')
        return(pipeline('ner', model=bert_model, tokenizer=bert_tokenizer))

    nlp = load_model()
    
    def extract_names(comment):
        """Extracts names from a text using a pre-trained NER model."""
        ner_list = nlp(comment)
        names = []
        current_name = []
        for ner_dict in ner_list:
            if ner_dict['entity'] == 'B-PER':
                if current_name:
                    names.append(' '.join(current_name).replace(' ##', '').replace(' .', '.'))
                current_name = [ner_dict['word']]
            elif ner_dict['entity'] == 'I-PER':
                current_name.append(ner_dict['word'])
        if current_name:
            names.append(' '.join(current_name).replace(' ##', '').replace(' .', '.'))
        return names
    
    # Apply the name extraction function
    df['Names'] = df['ACTUAL REVIEW'].apply(extract_names)

    # Explode the list of names into separate rows
    df = df.explode('Names')
    df = df.drop_duplicates()
    df = df.drop("REVIEWID", axis = 1)

    #Remove rows where Names is empty string or NaN
    df = df.dropna(subset=['Names'])
    df = df[df['Names'] != ""]

    def fix_broken_names(df):
        new_rows = []
        prev_row = None  # Store the previous row

        for _, row in df.iterrows():
            name = row["Names"]
            if name.startswith("##"):
                if prev_row is not None:  # Ensure there's a previous row
                    combined_name = prev_row["Names"] + name[2:]
                    new_row = prev_row.copy()
                    new_row["Names"] = combined_name
                    new_rows.append(new_row)
                    prev_row = None  # Clear the previous row
            else:
                if prev_row is not None:
                    new_rows.append(prev_row)
                prev_row = row.copy()  # Store the current row as the previous

        if prev_row is not None:  # Append the last row if it wasn't used
            new_rows.append(prev_row)

        return pd.DataFrame(new_rows)

    fixed_df = fix_broken_names(df)

    # Group by property and name, calculate average star rating
    grouped = fixed_df.groupby(['PROPERTY NAME', 'Names'])['STAR RATING'].agg(['mean', 'count']).reset_index()
    grouped = grouped.rename(columns={'mean': 'Average Star Rating', 'count': 'Review Count'})

    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')

    st.download_button(
            label="Download the Raw output file", 
            data=convert_df_to_csv(fixed_df), 
            file_name="raw.csv",
            mime="text/csv"
    )
    st.download_button(
            label="Download the analysis file", 
            data=convert_df_to_csv(grouped), 
            file_name="analysis.csv",
            mime="text/csv"
    )
