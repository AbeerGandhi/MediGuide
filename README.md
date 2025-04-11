# 🩺 MediGuide
## A Deep Learning based Alternate Medicine Recommendation System and SBERT-based intelligent medicine recommendation system.

## MediGuide helps identify alternative medicines based on their composition, chemical and action classes. It leverages both traditional deep learning (BiLSTM) and transformer-based (SBERT) models to semantically understand medicine properties and recommend substitutes. This can assist in medicine substitution, especially in cases of unavailability or cost sensitivity.

# 🔹 Model 1: BiLSTM-Based Alternate Medicine Recommendation

**Method Used:** Custom BiLSTM-based embedding generation + Cosine Similarity

**Dataset Used:**
medicine_dataset.csv

# 🔹 Model 2: SBERT-Based Medicine Recommendation

**Method Used:** Sentence-BERT embeddings + Cosine Similarity

**Dataset Used:**
drugsComTest_raw.csv
