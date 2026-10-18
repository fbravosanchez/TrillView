# -*- coding: utf-8 -*-
#%% notes
"""
Created by Francisco Bravo Sanchez Oct 2026

Generates birdset labels


"""


import requests
import pandas as pd

def get_taxonomy():
    taxonomy_url = "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv"
    taxonomy = pd.read_csv(taxonomy_url)
    return taxonomy

def get_dataset(url):
    response = requests.get(url)
    mapping = response.json()
    df = pd.DataFrame(mapping["label2id"].items(),
        columns=["ebird_code", "class_id"])
    df = df.merge(taxonomy[["SPECIES_CODE", "SCIENTIFIC_NAME", "COMMON_NAME"]],
        left_on="ebird_code", right_on="SPECIES_CODE", how="left").drop(columns="SPECIES_CODE")
    return df


taxonomy = get_taxonomy()

hsn = get_dataset("https://raw.githubusercontent.com/DBD-research-group/BirdSet/main/resources/ebird_codes/HSN_ebird_codes.json")
ssw = get_dataset("https://raw.githubusercontent.com/DBD-research-group/BirdSet/main/resources/ebird_codes/SSW_ebird_codes.json")

hsn.to_csv('/data/hsn_labels.csv', index=False)
ssw.to_csv('/data/ssw_labels.csv', index=False)
