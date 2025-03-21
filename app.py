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
import asyncio

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

import openai
from openai import AsyncOpenAI

# Case 1: OpenAI を使った ０ からの生成関数
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
async def generate_story(prompt, model="gpt-3.5-turbo", max_tokens=300):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens, 
            temperature=0.8,
        )
        # Access the content in the latest response format
        story = response.choices[0].message.content
        return story
        
    except Exception as e:
        return f"An error occurred: {e}"

# Case 2: T5 による新しい Summary 生成関数
def generate_summary_from_multiple_docs(docs, prefix="create a coherent story summary: "):
    combined_text = " ".join(docs)
    input_text = prefix + combined_text
    inputs = tokenizer_t5(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    with torch.no_grad():
        output_ids = model_t5.generate(
            **inputs,
            min_length=100,
            max_length=300,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=False
        )
    return tokenizer_t5.decode(output_ids[0], skip_special_tokens=True)

# OpenAI API を使って Summary を自然な文章にする関数
async def refine_summary_with_openai(summary):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at writing natural and engaging summaries."},
            {"role": "user", "content": f"Please refine the following summary to make it more natural and engaging:\n\n{summary}"}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


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

    # Switch はここで
    if not summaries:
        print("該当なし (新しい Summary を生成します)")
        #ai_answer = asyncio.run(generate_story(query))
        return jsonify({"error": "ナッシング！！"})
    else:
        #T5_answer = generate_summary_from_multiple_docs(summaries)
        print("\n 近似 5 件の類似 Summary を元に新しい Summary を生成しました")
        '''
        #print("\n T5 が生成した Summary:")
        #print(T5_answer)
        '''
        #ai_answer = asyncio.run(refine_summary_with_openai(T5_answer))
        return jsonify({"error": "ありりんご！"})


@app.route("/", methods=["GET"])
def get_summary(): 
    query = request.args.get("query", default="genre: fantasy, summary: A young girl, Miu starts school and meets a special friend.")
    TARGET_SIMILARITY = float(request.args.get("TARGET_SIMILARITY", 0.4))
    SIMILARITY_THRESHOLD = float(request.args.get("SIMILARITY_THRESHOLD", 0.1))
    
    ai_answer = process_query(query, TARGET_SIMILARITY, SIMILARITY_THRESHOLD)
    return ai_answer
    #return jsonify({"result": ai_answer})

'''
def run_async_function(async_func, *args):
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.ensure_future(async_func(*args))
        return loop.run_until_complete(future)
    except RuntimeError:
        return asyncio.run(async_func(*args))
    ai_answer = run_async_function(refine_summary_with_openai, T5_answer)
'''

@app.route("/summary")
def index():
    return jsonify({"message": "Welcome to the Plot Generation API!"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
