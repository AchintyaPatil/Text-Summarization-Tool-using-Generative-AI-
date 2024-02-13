import streamlit as st
import streamlit.components.v1 as components
with open('unique.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
from flask import Flask, request, jsonify
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import mammoth
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer


# st.components.v1.html(index.html, width=None, height=None, scrolling=False)


# >>> import plotly.express as px
# >>> fig = px.box(range(10))
# >>> fig.write_html('test.html')

# st.header("test html import")

HtmlFile = open("index.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code)

# path_to_html = "./index.html" 

# # Read file and keep in variable
# with open(path_to_html,'r') as f: 
#     html_data = f.read()

app = Flask(__name__)
CORS(app)

# Load Flan-T5 model
flan_t5_model_id = "google/flan-t5-small"
flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_id)
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_id)

def extract_text_from_docx_with_mammoth(file):
    result = mammoth.extract_raw_text(file)
    return result.value

def sentence_similarity(sent1, sent2):
    vectorizer = CountVectorizer().fit_transform([sent1, sent2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    return similarity_matrix

def textrank_summary(text, num_sentences=5):
    sentences = sent_tokenize(text)
    similarity_matrix = build_similarity_matrix(sentences)
    scores = np.array([sum(similarity_matrix[i]) for i in range(len(sentences))])
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1][:num_sentences]]
    return ' '.join(ranked_sentences)

def summarize_with_flan_t5(text):
    inputs = flan_t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = flan_t5_model.generate(inputs, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    return flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def combine_summaries(summary1, summary2):
    # Combine summaries using concatenation
    return summary1 + " " + summary2

def calculate_rouge_scores(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

@app.route('/process-text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        try:
            docx_file = request.files['file']
            extracted_text = extract_text_from_docx_with_mammoth(docx_file)

            # Generate summaries using Flan-T5 and TextRank
            flan_t5_summary = summarize_with_flan_t5(extracted_text)
            tr_summary = textrank_summary(extracted_text)  # Rename the variable

            # Combine summaries using your preferred method
            combined_summary = combine_summaries(flan_t5_summary, tr_summary)

            # Calculate ROUGE scores
            tr_scores = calculate_rouge_scores(extracted_text, tr_summary)
            combined_scores = calculate_rouge_scores(extracted_text, combined_summary)

            # Print the ROUGE scores
            print("TextRank ROUGE Scores:", tr_scores)
            print("Combined ROUGE Scores:", combined_scores)

            # Respond with the combined summary
            response_data = {
                'processedText': combined_summary,
                'rougeScores': {
                    'TextRank': tr_scores,
                    'Combined': combined_scores
                }
            }
            return jsonify(response_data)
        except Exception as e:
            print("Error:", str(e))
            return jsonify({'error': 'An error occurred during text processing.'})
    else:
        return jsonify({'message': 'Endpoint is accessible.'})

if __name__ == '__main__':
    app.run()
