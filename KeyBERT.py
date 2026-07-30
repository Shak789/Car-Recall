import pandas as pd
from keybert import KeyBERT
from tqdm import tqdm
import re

complaint = pd.read_csv("complaint.csv")

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

complaint['short_text'] = complaint['summary_clean'].str.slice(0, 2000)

results = []

docs = complaint['summary_clean'].str.slice(0, 500).tolist()

print("Starting extraction...")
for text in tqdm(complaint['short_text']):
    try:
        kp = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=5
        )

        words = [val[0] for val in kp]
        results.append(", ".join(words))
    except:
        results.append("Error in processing")

complaint['top_words'] = results
print("Done!")
print(complaint[['top_words']].head())

complaint.to_csv("words.csv")
