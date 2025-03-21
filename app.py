import os
from dotenv import load_dotenv
import asyncio
from flask import Flask, jsonify, request 

load_dotenv()  # .env の読み込み
app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHEET_ID = os.getenv("SHEET_ID")
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"


import faiss
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 公開された Google スプレッドシートの 読み込み
df = pd.read_csv(CSV_URL)
print("Googleスプレッドシートからデータを取得しました！")

# Embedding　モデルのロード（検索用）
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# T5 (生成) モデルのロード
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
model_t5.to("cpu")

# データベースを作成
docs = df.apply(lambda row: ",  ".join(f"{col}: {val}" for col, val in zip(df.columns, row)), axis=1).tolist()
'''
# 埋め込み生成
doc_embeddings = embedding_model.encode(docs, batch_size=64, show_progress_bar=True)
'''
# Railway のボリュームに保存したファイルのパス
FILE_PATH = "./doc_embeddings.npy"
# ファイルが存在するか確認
if os.path.exists(FILE_PATH):
    doc_embeddings = np.load(FILE_PATH)
    print("doc_embeddings.npy をロードしました！")
else:
    print("エラー: doc_embeddings.npy が見つかりません！")
    

# ユーザーの質問を受け取る
def process_query(query, TARGET_SIMILARITY, SIMILARITY_THRESHOLD):
    #query_embedding = embedding_model.encode([query])
    query_embedding = np.array(embedding_model.encode([query])).astype('float32')

    # FAISS ベクトル検索エンジンを構築
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    # FAISS を使って類似文書を検索 (上位5件)
    D, I = index.search(query_embedding, k=5)

    # コサイン類似度を計算
    query_vector = query_embedding / np.linalg.norm(query_embedding)  # 正規化
    doc_vectors = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)  # 正規化
    similarities = cosine_similarity(query_vector, doc_vectors)[0]

    # ターゲット類似度に最も近い文書を取得
    closest_docs = [(docs[i], similarities[i]) for i in range(len(docs))]
    sorted_docs = sorted(closest_docs, key=lambda x: abs(x[1] - TARGET_SIMILARITY))[:5]
    '''
    # 上位5件の Summary を表示
    print(f"\n 質問: {query}\n")
    print("上位5件の類似文書 (TARGET_SIMILARITY に最も近いものを選択):\n")
    '''

    summaries = []
    for doc, sim in sorted_docs:
        doc_data = dict(item.split(": ", 1) for item in doc.split(",  ") if ": " in item)
        if abs(sim - TARGET_SIMILARITY) <= SIMILARITY_THRESHOLD:
            title_text = doc_data.get("title", "No title available")
            genre_text = doc_data.get("genre", "No genre available")
            summary_text = doc_data.get("summary", "No summary available")
            summaries.append(summary_text)
            '''
            print(f" 類似度: {sim:.2f} | Title: {title_text} | Genre: {genre_text}")
            print(f" Summary: {summary_text}\n")
            '''

    test = []
    test.append(query)
    test.append(TARGET_SIMILARITY)
    test.append(SIMILARITY_THRESHOLD)
    return test

    
@app.route("/", methods=["GET"])
def get_summary(): 
    query = request.args.get("query", default="genre: fantasy, summary: A young girl, Miu starts school and meets a special friend.")
    TARGET_SIMILARITY = float(request.args.get("TARGET_SIMILARITY", 0.4))
    SIMILARITY_THRESHOLD = float(request.args.get("SIMILARITY_THRESHOLD", 0.1))

    ai_answer = process_query(query, TARGET_SIMILARITY, SIMILARITY_THRESHOLD)
    return jsonify({"result": ai_answer})


@app.route("/summary")
def index():
    return jsonify({"message": "Welcome to the Plot Generation API!"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
