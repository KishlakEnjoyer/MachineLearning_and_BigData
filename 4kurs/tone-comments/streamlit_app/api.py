from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import mysql.connector
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

load_dotenv()

app = FastAPI(title="Comment Analysis API")

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "comments_db"),
        port=int(os.getenv("DB_PORT", 3306))
    )

MAX_LEN = 80

class MLModels:
    def __init__(self):
        self.model_bin = load_model("lstm_binary.keras")
        self.model_mul = load_model("lstm_multiclass.keras")
        with open("tokenizer_bin.pkl", "rb") as f:
            self.tokenizer_bin = pickle.load(f)
        with open("tokenizer_mult.pkl", "rb") as f:
            self.tokenizer_mul = pickle.load(f)

models = MLModels()

def preprocess(text, tokenizer):
    text = text.lower().replace("ё", "е")
    seq = tokenizer.texts_to_sequences([text])
    return sequence.pad_sequences(seq, maxlen=MAX_LEN)

class CommentInput(BaseModel):
    comment_text: str
    save_to_db: bool = True

class AnalysisResponse(BaseModel):
    comment_text: str
    is_toxic: bool
    toxic_probability: float
    toxic_label: str
    category: str
    confidence: float
    all_probabilities: dict
    comment_id_toxic: Optional[int] = None
    comment_id_multiclass: Optional[int] = None

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_comment(data: CommentInput):
    x_bin = preprocess(data.comment_text, models.tokenizer_bin)
    toxic_prob = float(models.model_bin.predict(x_bin, verbose=0)[0][0])
    is_toxic = toxic_prob > 0.5
    
    x_mul = preprocess(data.comment_text, models.tokenizer_mul)
    mul_probs = models.model_mul.predict(x_mul, verbose=0)[0]
    classes = ["normal", "insult", "threat", "obscenity"]
    top_idx = int(np.argmax(mul_probs))
    category = classes[top_idx]
    confidence = float(mul_probs[top_idx])
    
    one_hot = {cls: 0 for cls in classes}
    one_hot[category] = 1
    
    comment_id_toxic = None
    comment_id_multiclass = None
    
    if data.save_to_db:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO comments_toxic (comment_text, comment_toxic)
                VALUES (%s, %s)
            """, (data.comment_text, 1 if is_toxic else 0))
            comment_id_toxic = cursor.lastrowid
            
            cursor.execute("""
                INSERT INTO comments_multiclass 
                (comment_text, normal, insult, threat, obscenity)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                data.comment_text,
                one_hot["normal"],
                one_hot["insult"],
                one_hot["threat"],
                one_hot["obscenity"]
            ))
            comment_id_multiclass = cursor.lastrowid
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")
    
    return AnalysisResponse(
        comment_text=data.comment_text,
        is_toxic=is_toxic,
        toxic_probability=toxic_prob,
        toxic_label="Токсичный" if is_toxic else "Не токсичный",
        category=category.capitalize(),
        confidence=confidence,
        all_probabilities={cls: float(prob) for cls, prob in zip(classes, mul_probs)},
        comment_id_toxic=comment_id_toxic,
        comment_id_multiclass=comment_id_multiclass
    )

@app.get("/api/history/toxic")
async def get_toxic_history(limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM comments_toxic ORDER BY comment_id DESC LIMIT %s", (limit,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"count": len(results), "comments": results}

@app.get("/api/history/multiclass")
async def get_multiclass_history(limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM comments_multiclass ORDER BY comment_id DESC LIMIT %s", (limit,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return {"count": len(results), "comments": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)