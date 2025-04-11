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

#  Step 1: Load Dataset
df = pd.read_csv("Datasets\medicine_dataset.csv")
df.fillna("", inplace=True)  # Koi missing value hai toh usko khali string se replace karo

#  Step 2: Medicine ka text feature bana rahe hain
df["medicine_text"] = df["name"] + " " + df["Chemical Class"] + " " + df["Action Class"]

#  Step 3: Tokenization & Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # Sirf 5000 words tak rakho, baaki <OOV>
tokenizer.fit_on_texts(df["medicine_text"])  # Medicine text ko tokenize karo
sequences = tokenizer.texts_to_sequences(df["medicine_text"])  # Tokens me convert karo
max_len = max(len(seq) for seq in sequences)  # Sabse lambi sequence ka length nikalo
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")  # Pura sequence ek size ka banao

#  Step 4: BiLSTM-Based Deep Learning Model Define karna
class MedicineBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MedicineBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Words ke embeddings create kar rahe hain
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)  # BiLSTM layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # LSTM ke output ko fully connected layer se pass karo

    def forward(self, x, lengths):
        x = self.embedding(x)  # Token ko embedding vector me convert karo
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)  # Padded sequence ko pack karo
        _, (hidden, _) = self.lstm(x)  # LSTM ka final hidden state nikalo
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Dono direction ka hidden state concatenate karo
        return self.fc(hidden)  # Final output generate karo

#  Step 5: Model Parameters Set Karna
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size jitne unique words hain uske hisaab se set karo
embedding_dim = 128  # Embedding vector ka size
hidden_dim = 32  # LSTM ka hidden layer size
output_dim = 64  # Final output dimension, jo cosine similarity ke liye use hoga

#  Step 6: Model Initialize Karna
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU available hai toh use karo, nahi toh CPU
model = MedicineBiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)  # Model ko device pe send karo

#  Step 7: Data Ko PyTorch Tensor Me Convert Karo
padded_sequences = torch.tensor(padded_sequences, dtype=torch.long).to(device)  # Sequence tensor banaya
sequence_lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long).to(device)  # Sequence ka actual length store kiya

#  Step 8: Medicine Embeddings Generate Karo
with torch.no_grad():
    medicine_embeddings = model(padded_sequences, sequence_lengths).cpu().numpy()  # LSTM model ka output le lo embeddings ke form me

#  Step 9: Similar Medicines Dhoondhna
def get_substitutes(medicine_name, df, tokenizer, model, medicine_embeddings):
    if medicine_name not in df["name"].values:
        return f" Error: Medicine '{medicine_name}' not found in the database.\n"

    idx = df[df["name"] == medicine_name].index[0]  # Given medicine ka index dhundo

    #  Query ke liye Embedding Generate Karo
    query_text = df.iloc[idx]["medicine_text"]
    query_seq = tokenizer.texts_to_sequences([query_text])  # Query text ko tokenize karo
    query_seq = pad_sequences(query_seq, maxlen=max_len, padding="post")  # Padding apply karo

    query_tensor = torch.tensor(query_seq, dtype=torch.long).to(device)
    query_length = torch.tensor([len(query_text.split())], dtype=torch.long).to(device)

    with torch.no_grad():
        query_embedding = model(query_tensor, query_length).cpu().numpy()  # Query ka embedding nikal lo

    #  Cosine Similarity Calculate Karo
    similarity_scores = cosine_similarity(query_embedding, medicine_embeddings)[0]  # Query vs All Medicines
    similar_indices = np.argsort(similarity_scores)[::-1][1:]  # Top similar medicines dhoondo

    #  Direct Substitutes (Agar database me diye gaye hain toh max 3 lo)
    direct_substitutes = [
        df.iloc[idx][f"substitute{i}"]
        for i in range(5) if df.iloc[idx][f"substitute{i}"]
    ][:3]

    #  Deep Learning Model ke Top Alternatives (Max 5 lo aur repeat avoid karo)
    recommended_medicines = []
    recommended_scores = []
    for idx in similar_indices:
        if len(recommended_medicines) >= 7:  # Max 7 recommended medicines de sakte hain
            break
        med_name = df.iloc[idx]["name"]
        if med_name not in direct_substitutes:  # Agar direct substitute me already hai toh skip karo
            recommended_medicines.append(med_name)
            recommended_scores.append(similarity_scores[idx])

    #  Final List Banake Ensure Karo ki Koi Medicine Repeat Na Ho
    final_recommendations = list(dict.fromkeys(direct_substitutes + recommended_medicines))
    final_scores = [similarity_scores[df[df["name"] == med].index[0]] for med in final_recommendations if med in df["name"].values]

    #  Output ko Properly Format Karo
    output = f"\n Medicine Name: {medicine_name}\n"
    output += "=" * 50 + "\n"

    if final_recommendations:
        output += " Recommended Alternatives:\n"
        for idx, (med, score) in enumerate(zip(final_recommendations, final_scores), start=1):
            output += f"   {idx}. {med} (Similarity Score: {score:.4f})\n"
    else:
        output += "No similar medicines found.\n"

    return output

#  Step 10: User Input Le Kar Substitute Medicines Dhoondo
medicine_name = input(" Enter medicine name: ")
print(get_substitutes(medicine_name, df, tokenizer, model, medicine_embeddings))
