from flask import Flask, render_template, request, jsonify
from gpt_model import generate_text

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"response": "Please enter a prompt!"})
    
    response = generate_text(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)