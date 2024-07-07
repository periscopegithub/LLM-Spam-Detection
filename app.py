from flask import Flask, request, render_template, current_app
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)


def load_model():
    checkpoint_path = "./experiments/checkpoint-26835"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return model, tokenizer


model, tokenizer = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form.get("email", "").strip()
        if not email_text:
            return render_template(
                "classifier.html",
                classification="No email text provided. Please paste an email message.",
            )
        classification = classify_email(email_text, model, tokenizer)
        return render_template("classifier.html", classification=classification)
    return render_template("classifier.html", classification=None)


def classify_email(text, model, tokenizer):
    prefix = "classify as ham or spam: "
    encoded_input = tokenizer(
        prefix + text, return_tensors="pt", truncation=True, max_length=512
    )
    output = model.generate(**encoded_input, max_length=5)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.strip()


if __name__ == "__main__":
    app.run(debug=True)
