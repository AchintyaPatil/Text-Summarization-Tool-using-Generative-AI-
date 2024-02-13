import requests
from flask import Flask, request, jsonify
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import mammoth
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

@app.route('/process-text', methods=['POST'])
def process_text():
    if request.method == 'POST':
        try:
            docx_file = request.files['file']
            extracted_text = extract_text_from_docx_with_mammoth(docx_file)

            # Log the extracted text to the console
            print("Extracted Text:", extracted_text)

            # Perform TextRank-based summarization
            summarized_text = textrank_summary(extracted_text)

            # Log the summarized text to the console
            print("Summarized Text:", summarized_text)

            # Respond back to the JavaScript frontend with the processed text
            response_data = {
                'processedText': 'Summarized Text: ' + summarized_text  # Modify this line based on your processing logic
            }
            return jsonify(response_data)
        except Exception as e:
            # Handle any errors that might occur during text extraction or processing
            print("Error:", str(e))
            return jsonify({'error': 'An error occurred during text processing.'})
    else:
        # Handle GET request (if needed)
        # For example, you can return a message indicating that the endpoint is accessible
        return jsonify({'message': 'Endpoint is accessible.'})

if __name__ == '__main__':
    app.run()
