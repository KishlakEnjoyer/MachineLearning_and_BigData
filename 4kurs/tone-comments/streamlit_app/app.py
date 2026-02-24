import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Анализ комментариев", layout="centered")
st.title("💬 Анализ комментариев")

# Ввод комментария
comment = st.text_area("Введите комментарий:", height=100)
save_to_db = st.checkbox("💾 Сохранить в БД", value=True)

if st.button("🔮 Предсказать"):
    if not comment.strip():
        st.warning("Введите текст комментария!")
    else:
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json={"comment_text": comment, "save_to_db": save_to_db},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.subheader("📊 Результаты")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Токсичность", data["toxic_label"], f"{data['toxic_probability']:.2%}")
                
                with col2:
                    st.metric("Категория", data["category"], f"{data['confidence']:.2%}")
                
                st.subheader("Вероятности по классам")
                for cls, prob in data["all_probabilities"].items():
                    st.write(f"{cls.capitalize()}: {prob:.2%}")
                    st.progress(prob)
                
                if data["is_toxic"]:
                    st.error("⚠ Комментарий токсичный!")
                else:
                    st.success("✅ Комментарий безопасен")
                
                if save_to_db:
                    st.info(f"✅ Сохранено: toxic_id={data['comment_id_toxic']}, multiclass_id={data['comment_id_multiclass']}")
            else:
                st.error(f"Ошибка API: {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка подключения: {str(e)}")