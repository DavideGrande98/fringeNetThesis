#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:10:58 2023
This code performs the sentiment analysis over a csv file containing strings with the users-written content of the post/comment 
@author: davidegrande
"""

#%%
from detoxify import Detoxify
import pandas as pd
import torch
import matplotlib.pyplot as plt
import re

def cleanTxt(text):
 text=text.lower()
 text = re.sub('\S*trump\S*', 'trump', text)
 text = re.sub('\S*biden\S*', 'biden', text)
 text = re.sub('@\S+', '', text) #Rimuove le @menzioni
 text = re.sub('#', '', text) # Rimuove l'hashtag
 text = re.sub('\S*black lives matter\S*', 'blm', text)
 text = re.sub('\S*make america great again\S*', 'maga', text)
 
 return text



df = pd.read_csv('/Users/davidegrande/Desktop/Dataset_Tesi/csv_files/body_post.csv', low_memory=False)
df = df.dropna(subset=['body'])

df['body'] = df['body'].apply(cleanTxt)

#%%
body_list = df['body'].tolist()
words_to_filter = ['biden','trump','maga', 'blm']
filtered_lists = {}
for word in words_to_filter:
    filtered_lists[word] = [s for s in body_list if word in s.lower()]
#%%
results = Detoxify('multilingual',  device=torch.device('cpu'))
#%%
# for k in filtered_lists:
#     filtered_lists[k] = filtered_lists[k][0:100]
#     res = results.predict(filtered_lists[k])
#     df_analysis = pd.DataFrame.from_dict(res)
#     plt.clf()
#     df_analysis.plot()
#     plt.savefig('/Users/davidegrande/Dropbox (Politecnico Di Torino Studenti)/Magistrale/Tesi_Grande/spyder/Plots/sentiment_analysis/Content_scores_hashtag_{}.png'.format(k), dpi=300)
    
 
#%%
means = dict()
for k in filtered_lists:
    filtered_lists[k] = filtered_lists[k][0:200]
    res = results.predict(filtered_lists[k])
    df_analysis = pd.DataFrame.from_dict(res)
    for c in df_analysis.columns:
        means[c] = df_analysis[c].mean()
        plt.clf()
        plt.bar(means.keys(),means.values())
        plt.savefig('/Users/davidegrande/Dropbox (Politecnico Di Torino Studenti)/Magistrale/Tesi_Grande/spyder/Plots/sentiment_analysis/barplot_{}.png'.format(k), dpi=300)



