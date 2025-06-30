import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("model.h1.24_jan_19.keras")

# Load tokenizers
with open("tokenizer_de.pkl", "rb") as f:
    tokenizer_de = pickle.load(f)
with open("tokenizer_en.pkl", "rb") as f:
    tokenizer_en = pickle.load(f)

# Set max input length (ubah sesuai model kamu)
max_input_len = 10

# Fungsi prediksi
def translate(sentence):
    seq = tokenizer_de.texts_to_sequences([sentence.lower()])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')
    preds = model.predict(padded)
    output_seq = np.argmax(preds[0], axis=1)
    result = tokenizer_en.sequences_to_texts([output_seq])
    return result[0]

# Streamlit UI
st.title("🇩🇪➡️🇺🇸 Penerjemah Jerman ke Inggris")
input_text = st.text_input("Masukkan kalimat Bahasa Jerman:")

if st.button("Terjemahkan"):
    if input_text.strip() != "":
        hasil = translate(input_text)
        st.success(f"**Hasil Terjemahan:** {hasil}")
    else:
        st.warning("Masukkan kalimat terlebih dahulu.")
