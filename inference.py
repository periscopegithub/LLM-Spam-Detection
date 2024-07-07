import tkinter as tk
from tkinter import messagebox, scrolledtext
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Use a raw string or forward slashes for file paths
checkpoint_path = r"./experiments/checkpoint-26835"

try:
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
except Exception as e:
    print(f"An error occurred while loading the model or tokenizer: {e}")
    raise


def classify_email(text, model, tokenizer):
    # Prefix required by FLAN-T5 for classification tasks
    prefix = "classify as ham or spam: "

    # Prepare the text for the model
    encoded_input = tokenizer(
        prefix + text, return_tensors="pt", truncation=True, max_length=512
    )

    # Explicitly set a very conservative maximum generation length
    output = model.generate(
        **encoded_input, max_length=5
    )  # Enough for one word plus any potential special tokens

    # Decode the output to human-readable text
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result.strip()  # Strip whitespace to clean up the output


def main():
    root = tk.Tk()
    root.title("Email Classifier")

    # Create a scrollable Text widget
    txt = scrolledtext.ScrolledText(root, width=60, height=10)
    txt.pack(padx=10, pady=10)

    # Function to handle classification
    def on_classify():
        email_text = txt.get("1.0", "end-1c")  # Get text from Text widget
        if email_text:
            try:
                # Classify the email
                result = classify_email(email_text, model, tokenizer)
                # Display the result in a message box
                messagebox.showinfo(
                    "Classification Result", f"The email is classified as: {result}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred during classification: {e}"
                )
        else:
            messagebox.showinfo("No Input", "No email message was provided.")

    # Button to trigger classification
    btn_classify = tk.Button(root, text="Classify Email", command=on_classify)
    btn_classify.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
