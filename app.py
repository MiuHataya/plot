from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# AI モデルの準備
generator = pipeline("text-generation", model="gpt2")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return jsonify({"response": response[0]["generated_text"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
