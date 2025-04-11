import pandas as pd
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from google.colab import files

# Dataset load karo
df = pd.read_csv("Datasets\drugsComTest_raw.csv")

# Sirf relevant columns rakho
df = df[['drugName', 'condition', 'review', 'rating', 'usefulCount']]
df.dropna(subset=['condition'], inplace=True)  # Jo condition missing hai unko hatao

# SBERT Model initialize karo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sabhi conditions ke embeddings generate karo
conditions = df['condition'].unique().tolist()
condition_embeddings = model.encode(conditions, convert_to_tensor=True)

# Medicine recommend karne ka function
def recommend_medicine(user_condition):
    user_embedding = model.encode(user_condition, convert_to_tensor=True)  # User ke input ka embedding banao

    # Similarity scores calculate karo
    similarities = util.pytorch_cos_sim(user_embedding, condition_embeddings)[0]
    best_match_index = similarities.argmax()  # Sabse best match ka index nikalo
    best_match_condition = conditions[best_match_index]  # Best match wali condition lo

    # Matched condition ke related medicines dhoondo
    recommended_meds = df[df['condition'] == best_match_condition].copy()
    recommended_meds = recommended_meds.sort_values(by=['rating', 'usefulCount'], ascending=False)  # Rating aur usefulCount ke basis pe sort karo

    # Top 10 unique medicines nikalo (duplicate handle karne ke liye set use karo)
    seen_medicines = set()
    top_medicines = []

    for med in recommended_meds['drugName']:
        med_lower = med.lower()  # Case normalize karo
        if med_lower not in seen_medicines:
            seen_medicines.add(med_lower)
            top_medicines.append(med)
        if len(top_medicines) == 10:
            break  # Sirf 10 medicines tak ruko

    return best_match_condition, top_medicines

# Example usage
user_input = input("Describe your health condition or symptoms: ").strip()
best_match, recommended_medicines = recommend_medicine(user_input)

# Results display karo
print(f"\nMatched Condition: {best_match}")
print("\nTop Recommended Medicines:")

for med in recommended_medicines:
    print(f"- {med}")
