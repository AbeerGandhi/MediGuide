# =================== Common Imports ===================
import pandas as pd
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from warnings import filterwarnings
filterwarnings("ignore")

import gradio as gr
from sentence_transformers import SentenceTransformer, util

# =================== Load BiLSTM Dataset ===================
df_bilstm = pd.read_csv("medicine_dataset.csv")
df_bilstm.fillna("", inplace=True)

# Step 2: Medicine ka text feature bana rahe hain
df_bilstm["medicine_text"] = df_bilstm["name"] + " " + df_bilstm["Chemical Class"] + " " + df_bilstm["Action Class"]

# Step 3: Tokenization & Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_bilstm["medicine_text"])
sequences = tokenizer.texts_to_sequences(df_bilstm["medicine_text"])
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

# Step 4: BiLSTM-Based Deep Learning Model Define karna
class MedicineBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MedicineBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)

# Step 5: Model Parameters Set Karna
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 32
output_dim = 64

# Step 6: Model Initialize Karna
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bilstm = MedicineBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# Step 7: Data Ko PyTorch Tensor Me Convert Karo
padded_sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long).to(device)
sequence_lengths_tensor = torch.tensor([len(seq) for seq in sequences], dtype=torch.long).to(device)

# Step 8: Medicine Embeddings Generate Karo
with torch.no_grad():
    medicine_embeddings = model_bilstm(padded_sequences_tensor, sequence_lengths_tensor).cpu().numpy()

# Step 9: Similar Medicines Dhoondhna
def get_substitutes(medicine_name):
    if medicine_name not in df_bilstm["name"].values:
        return f"❌ Error: Medicine '{medicine_name}' not found in the database."

    idx = df_bilstm[df_bilstm["name"] == medicine_name].index[0]
    query_text = df_bilstm.iloc[idx]["medicine_text"]
    query_seq = tokenizer.texts_to_sequences([query_text])
    query_seq = pad_sequences(query_seq, maxlen=max_len, padding="post")

    query_tensor = torch.tensor(query_seq, dtype=torch.long).to(device)
    query_length = torch.tensor([len(query_text.split())], dtype=torch.long).to(device)

    with torch.no_grad():
        query_embedding = model_bilstm(query_tensor, query_length).cpu().numpy()

    similarity_scores = cosine_similarity(query_embedding, medicine_embeddings)[0]
    similar_indices = np.argsort(similarity_scores)[::-1][1:]

    direct_substitutes = [
        df_bilstm.iloc[idx][f"substitute{i}"]
        for i in range(5) if df_bilstm.iloc[idx][f"substitute{i}"]
    ][:3]

    recommended_medicines = []
    for sim_idx in similar_indices:
        med_name = df_bilstm.iloc[sim_idx]["name"]
        if med_name not in direct_substitutes and med_name not in recommended_medicines:
            recommended_medicines.append(med_name)
        if len(recommended_medicines) == 7:
            break

    final_recommendations = list(dict.fromkeys(direct_substitutes + recommended_medicines))
    final_scores = [similarity_scores[df_bilstm[df_bilstm["name"] == med].index[0]] for med in final_recommendations if med in df_bilstm["name"].values]

    output = f"\n🔍 **Medicine Name:** {medicine_name}\n"
    output += "📋 **Recommended Alternatives:**\n"
    for idx, (med, score) in enumerate(zip(final_recommendations, final_scores), start=1):
        output += f"{idx}. {med} (Similarity Score: {score:.4f})\n"
    return output

# =================== SBERT Section ===================
# Dataset load karo
df_sbert = pd.read_csv("drugsComTest_raw.csv")
df_sbert = df_sbert[['drugName', 'condition', 'review', 'rating', 'usefulCount']]
df_sbert.dropna(subset=['condition'], inplace=True)

# SBERT Model initialize karo
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sabhi conditions ke embeddings generate karo
conditions = df_sbert['condition'].unique().tolist()
condition_embeddings = sbert_model.encode(conditions, convert_to_tensor=True)

# Medicine recommend karne ka function
def recommend_medicine(user_condition):
    user_embedding = sbert_model.encode(user_condition, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, condition_embeddings)[0]
    best_match_index = similarities.argmax()
    best_match_condition = conditions[best_match_index]

    recommended_meds = df_sbert[df_sbert['condition'] == best_match_condition].copy()
    recommended_meds = recommended_meds.sort_values(by=['rating', 'usefulCount'], ascending=False)

    seen_medicines = set()
    top_medicines = []
    for med in recommended_meds['drugName']:
        med_lower = med.lower()
        if med_lower not in seen_medicines:
            seen_medicines.add(med_lower)
            top_medicines.append(med)
        if len(top_medicines) == 10:
            break

    output = f"\n🩺 **Matched Condition:** {best_match_condition}\n\n💊 **Top Recommended Medicines:**\n"
    for med in top_medicines:
        output += f"- {med}\n"
    return output

# =================== Gradio Interface ===================
with gr.Blocks(title="Medicine Recommendation System") as demo:
    gr.Markdown("# 🧪 Medicine Recommendation System")

    with gr.Tab("🔁 BiLSTM: Medicine Substitute Finder"):
        gr.Markdown("Enter a medicine name to find recommended alternatives based on chemical and action class.")
        med_input = gr.Textbox(label="Medicine Name")
        med_output = gr.Textbox(label="Recommended Substitutes")
        med_btn = gr.Button("Find Substitutes")
        med_btn.click(fn=get_substitutes, inputs=med_input, outputs=med_output)

    with gr.Tab("💬 SBERT: Symptom-Based Recommender"):
        gr.Markdown("Describe your condition or symptoms to get top-rated medicines based on real user reviews.")
        cond_input = gr.Textbox(label="Describe your health condition")
        cond_output = gr.Textbox(label="Recommended Medicines")
        cond_btn = gr.Button("Recommend Medicines")
        cond_btn.click(fn=recommend_medicine, inputs=cond_input, outputs=cond_output)

demo.launch()
