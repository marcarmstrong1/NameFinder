# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:41:44 2022

@author: Marcus
"""

#File Format Submit_Date, Property, Comment, Score, (Name)

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import easygui

df = pd.read_csv(easygui.fileopenbox())
df["ACTUAL REVIEW"].dropna(inplace = True)

df = df[['DATE','PROPERTY NAME','ACTUAL REVIEW', "STAR RATING"]]
df = df[df["STAR RATING"] > 3]

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
        y = y.append({"Name" : [full_name], "Index" : count}, ignore_index = True)
    
    count = count + 1
    
y.set_index("Index", inplace = True)
y['Name'] = [','.join(map(str, l)) for l in y['Name']]
df = df.merge(y, how = "left", left_index = True, right_index = True)
df["Name"] = df["Name"].fillna("0")
df = df.groupby(['DATE','PROPERTY NAME','ACTUAL REVIEW', "STAR RATING"])['Name'].apply(', '.join).reset_index()

for x in range(len(df["Name"])):
    df["Name"][x] = df["Name"][x].replace(", ##", "")

df.to_excel("output.xlsx")
