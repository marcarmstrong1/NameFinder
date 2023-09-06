import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

st.title("A helpful tool to find names within reviews")

rating = st.slider("Select a rating (Ex. 4 would mean 4 stars and up)", 1, 5)

data = st.file_uploader("Upload a CSV file", type=["csv"])

if data is not None:
    df = pd.read_csv(data)
    df["ACTUAL REVIEW"].dropna(inplace = True)

    df = df[['DATE','PROPERTY NAME','ACTUAL REVIEW', "STAR RATING"]]
    df = df[df["STAR RATING"] > rating]

    bert_tokenizer = AutoTokenizer.from_pretrained('dslim/bert-large-NER')
    bert_model = AutoModelForTokenClassification.from_pretrained('dslim/bert-large-NER')

    nlp = pipeline('ner', model=bert_model, tokenizer=bert_tokenizer)

    count = 0
    y = pd.DataFrame(columns = ["Name", "Index"])

    for comment in df["ACTUAL REVIEW"]:
        names_string = comment
        ner_list = nlp(names_string)
    
        this_name = []
        all_names_list_tmp = []
    
        for ner_dict in ner_list:
            if ner_dict['entity'] == 'B-PER':
                if len(this_name) == 0:
                    this_name.append(ner_dict['word'])
                else:
                    all_names_list_tmp.append([this_name])
                    this_name = []
                    this_name.append(ner_dict['word'])
            elif ner_dict['entity'] == 'I-PER':
                this_name.append(ner_dict['word'])
    
        all_names_list_tmp.append([this_name])
    
        final_name_list = []
        for name_list in all_names_list_tmp:
            full_name = ' '.join(name_list[0]).replace(' ##', '').replace(' .', '.')
            final_name_list.append([full_name])
            y = y._append({"Name" : [full_name], "Index" : count}, ignore_index = True)
    
        count = count + 1
    
    y.set_index("Index", inplace = True)
    y['Name'] = [','.join(map(str, l)) for l in y['Name']]
    df = df.reset_index()
    df = df.merge(y, how = "left", left_index = True, right_index = True)
    df["Name"] = df["Name"].fillna("0")
    df = df.groupby(['DATE','PROPERTY NAME','ACTUAL REVIEW', "STAR RATING"])['Name'].apply(', '.join).reset_index()

    for x in range(len(df["Name"])):
        df["Name"][x] = df["Name"][x].replace(", ##", "")

    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')

    st.download_button(
            label="Download the output file", 
            data=convert_df_to_csv(df), 
            file_name="output.xlsx",
            mime="text/csv"
    )
