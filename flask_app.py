from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline

app = Flask(__name__)

# Load the legal dictionary from an Excel file
legal_dict = pd.read_excel('DIC.xlsx')

# Convert the legal dictionary into a dictionary for easy lookup
legal_to_civilian = dict(zip(legal_dict['WORD'], legal_dict['MEANING']))

# Function to convert legal text to civilian language
def legal_to_civilian_language(legal_text):
    # Replace legal terms with civilian language equivalents
    for term, civilian_term in legal_to_civilian.items():
        legal_text = re.sub(r'\b{}\b'.format(re.escape(term)), civilian_term, legal_text)

    return legal_text

# Initialize models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model_1 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize/', methods=['POST'])
def summarize():
    request_data = request.json
    input_text = request_data['text']

    result_d = summarizer(input_text, max_length=130, min_length=30, do_sample=False)

    # Check if data is a list with at least one element
    if isinstance(result_d, list) and len(result_d) > 0:
        # Access the first element (in this case, there's only one element)
        first_element = result_d[0]

        # Check if the 'summary_text' key exists in the dictionary and if it's a string
        if 'summary_text' in first_element and isinstance(first_element['summary_text'], str):
            summary_text = first_element['summary_text']
            result = summary_text

        else:
            print("Value for 'summary_text' not found or not a string")
    else:
        print("Data is empty or not a list")

    return jsonify({"summary": result})

@app.route('/translate/', methods=['POST'])
def translate():
    request_data = request.json
    input_text = request_data['text']

    model_inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Translate from English to Hindi
    generated_tokens = model_1.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"])

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return jsonify({"translation": translation})

if __name__ == '__main__':
    app.run(debug=True)
