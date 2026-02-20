import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


st.set_page_config(
    page_title="Анализ комментариев",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("💬 Анализ комментариев")
st.write("Введите комментарий, чтобы проверить токсичность и определить тональность.")

@st.cache_resource
def load_models():
    model_bin = load_model("lstm_binary.keras")
    model_mul = load_model("lstm_multiclass.keras")
    
    with open("tokenizer_bin.pkl", "rb") as f:
        tokenizer_bin = pickle.load(f)
    with open("tokenizer_mult.pkl", "rb") as f:
        tokenizer_mul = pickle.load(f)
    
    return model_bin, model_mul, tokenizer_bin, tokenizer_mul

model_bin, model_mul, tokenizer_bin, tokenizer_mul = load_models()

MAX_LEN = 80

def preprocess(text, tokenizer):
    text = text.lower().replace("ё", "е")
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = sequence.pad_sequences(text_seq, maxlen=MAX_LEN)
    return text_pad

comment = st.text_area("Введите комментарий:", height=120)

predict_button = st.button("🔮 Предсказать")

if predict_button:
    if comment.strip() == "":
        st.warning("Пожалуйста, введите комментарий для анализа.")
    else:
        x_bin = preprocess(comment, tokenizer_bin)
        toxic_prob = model_bin.predict(x_bin)[0][0]
        toxic_label = "Токсичный" if toxic_prob > 0.5 else "Не токсичный"
        
        x_mul = preprocess(comment, tokenizer_mul)
        mul_probs = model_mul.predict(x_mul)[0]
        classes = ["Normal", "Insult", "Threat", "Obscenity"]
        top_idx = np.argmax(mul_probs)
        mul_label = classes[top_idx]
        mul_confidence = mul_probs[top_idx]
        
        st.subheader("Результаты анализа:")
        st.markdown(f"**Токсичность:** {toxic_label} ({toxic_prob:.2f})")
        st.markdown(f"**Тип комментария:** {mul_label} ({mul_confidence:.2f})")
        
        st.subheader("Вероятности по классам тональности:")
        for cls, prob in zip(classes, mul_probs):
            st.write(f"{cls}: {prob:.2f}")
            st.progress(float(prob))

        if toxic_prob > 0.5:
            st.error("⚠ Этот комментарий токсичный!")
        else:
            st.success("✅ Комментарий безопасен")
