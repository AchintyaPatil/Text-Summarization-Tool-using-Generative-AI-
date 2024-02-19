import streamlit as st
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
# import time
# import psutil
# import collections
# import promlib
import streamlit.components.v1 as components

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



# # Metrics for tracking processing times
# model_inference_time = Gauge(name="model_inference_time", documentation="Time taken for model inference")
# preprocessing_time = Gauge(name="preprocessing_time", documentation="Time spent on text preprocessing")
# postprocessing_time = Gauge(name="postprocessing_time", documentation="Time spent on summarizing output")

# # Metrics for tracking model performance
# rogue_score = Gauge(name="rogue_score", documentation="ROUGE score of generated summary")
# bleu_score = Gauge(name="bleu_score", documentation="BLEU score of generated summary")
# distinct_n_grams = Counter(name="distinct_n_grams", documentation="Number of distinct n-grams in summary")

# # Metrics for tracking user interaction
# input_text_length = Gauge(name="input_text_length", documentation="Length of input text in characters")
# summary_length = Gauge(name="summary_length", documentation="Length of generated summary in characters")
# num_summarization_requests = Counter(name="num_summarization_requests", documentation="Total number of summarization requests")

# # Example usage
# def process_text(text):
#     # Simulate preprocessing (replace with your actual code)
#     start_preprocessing = time.time()
#     preprocessed_text = text.lower().strip()
#     preprocessing_time.set(time.time() - start_preprocessing)

#     # Simulate model inference (replace with your actual model call)
#     start_inference = time.time()
#     summary = "This is a generated summary."  # Replace with actual summary from your model
#     model_inference_time.set(time.time() - start_inference)

#     # Simulate postprocessing (replace with your actual code)
#     start_postprocessing = time.time()
#     summary = f"Summary: {summary}"  # Add formatting or context
#     postprocessing_time.set(time.time() - start_postprocessing)

#     # Calculate and set performance metrics (replace with actual calculation)
#     rogue_score.set(0.5)  # Replace with actual ROUGE score
#     bleu_score.set(0.3)  # Replace with actual BLEU score
#     distinct_n_grams.inc(len(set(summary.split())))

#     return summary

# if st.button("Summarize Text"):
#     input_text = st.text_area("Enter text to summarize:")
#     start_request = time.time()
#     if input_text:
#         input_text_length.set(len(input_text))
#         summary = process_text(input_text)
#         summary_length.set(len(summary))
#         st.success(summary)
#     num_summarization_requests.inc()
#     end_request = time.time()
#     # Additional metrics for debugging or analysis (optional)
#     # total_request_time.set(end_request - start_request)


# # st.components.v1.html(index.html, width=None, height=None, scrolling=False)


# # >>> import plotly.express as px
# # >>> fig = px.box(range(10))
# # >>> fig.write_html('test.html')

# st.header("test html import")
components.html(
    """
    <!DOCTYPE html>
<html lang="en" translate="no">
<head>
    <meta charSet="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    
    <link rel="stylesheet" href="unique.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">


    <script src="https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.0.0/mammoth.browser.min.js"></script>
    <title>Text Summarizer</title>
    <meta name="next-head-count" content="4" />

    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap');
body {
 
    font-weight: normal;
    font-style: normal;
  margin: 0;
  padding: 0;
  color: whitesmoke;
  
}

.header {
  margin: 36px 0 36px;
  display: flex;
  flex-direction: column;
  align-items: center;
}


*, *::before, *::after { box-sizing:border-box }

body {
	margin:0;
  padding: 0;
  background:#0d0d0d;
}



.title {
  font-weight: 600;
  font-size: 46px;
  text-align: center;
  padding: 0;
  margin: 0;
  margin-bottom: 12px;
}

.description {
  text-align: center;
  display: block;
  margin-bottom: 0;
  font-size: 20px;
  max-width: 530px;
}

.upload-container {
  max-width: 993px;
  margin: 0 auto;
  border-radius: 12px;
}

.upload-wrapper {
  padding: 8px;
}

.upload-btn {
  margin-top: -10px;
  margin-bottom: 22px;
  font-weight: normal;
}

.upload-drag-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.upload-drag-icon {
  clear: both;
}

.upload-text {
  font-weight: normal;
  margin-bottom: 22px;
}



.features {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  position: absolute;
  bottom: -7%;
  left: 280px;
  z-index: 300;
}

.feature {
  width: 323px;
  height: 200px; 
  margin-right: 12px;
  border: 1px solid #ccc;
  padding: 20px;
  transition: transform 0.2s, box-shadow 0.2s;
 /* margin-bottom: 4%; */
}

.feature:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.feature h2 {
  display: flex;
  gap: 8px;
  align-items: center;
  margin: 4px 0 16px;
}

.feature-description {
  text-align: justify;
}

















.major_text{
  position: relative;
  bottom: 110px;
}



.additional-features {
  display: flex;
  flex-direction: row; /* Display additional features in a row */
  justify-content: center; /* Center them horizontally */
  align-items: center;
}

.additional-feature {
  width: 323px;
  margin-right: 12px; /* Add margin between additional features */
}

.additional-feature:last-child {
  margin-right: 0; /* Remove margin for the last additional feature */
}



/* drag drop system */

.form {
  width: 500px;
  margin: 5% auto; 
  z-index: 10;
}

.form__container {
  position: relative;
  width: 100%;
  height: 200px;
  border: 2px dashed silver;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 18px;
  color: silver;
  margin-bottom: 5px;
  left: -280px;
}

.form__container.active {
  background-color: rgba(255,255,255,0.2); 
}

.form__file {
  position: relative;
  width: 100%;
  height: 200px;
  top: -206px;
  left: -280px;
  cursor: pointer;
  opacity: 0;
}

.form__files-container {
  display: flex;
  width: 100%;
  padding: 5px 0;
  justify-content: space-between;
  align-items: center;
}

.form__text {
  font-size: 18px;
  color: #333;
  max-width: 450px;
  
  overflow: hidden;
  text-overflow: ellipsis; 
}

.form__icon {
  font-size: 22px;
  color: #1871b5;
  text-decoration: none;
}

.text-bar{
  
    position: relative;
    width: 33%;
    height: 200px;
    border: 2px solid silver;
    bottom: 482px;
    font-size: 18px;
    
    margin-bottom: 5px;
    left: 844px;
    z-index: 300;
  
}

.summary-bar{
  
  position: relative;
  width: 43%;
  height: 300px;
  border: 2px solid silver;
  bottom: 415px;
  font-size: 18px;
  
  margin-bottom: 5px;
  left: 459px;
  z-index: 300;

}








.words{

  /* z-index: 300; */
  background-color: inherit;
  width: 100%;
  height: 100%;
  color: silver;
  font-size: large;
  text-align: center;
}







/* background stuff  */

@font-face {
  font-family: "Mona Sans";
  src: url("https://assets.codepen.io/64/Mona-Sans.woff2") format("woff2 supports variations"),
       url("https://assets.codepen.io/64/Mona-Sans.woff2") format("woff2-variations");
  font-weight: 100 1000;
}

@property --bg-1-x {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;
}

@property --bg-2-x {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;
}

@property --bg-2-y {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;
}

@property --bg-3-x {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;
}

@property --bg-3-y {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;  
}

:root {
  --bg-color: hsl(240deg 10% 12%);
  --bg-grain: url("https://assets.codepen.io/64/svgNoise2.svg");
  --bg-grain: url("data:image/svg+xml,%3Csvg viewBox='0 0 600 600' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");

  --shadow-size: max(140px, 40vw);
  --shadow-size-fallback: 40vw;
  --shadow-blur: 60;

  --color-1: #6328da;
  --color-2: #ff1bf1;
  --color-3: #008cea;

  --bg-1-x: 0;
  --bg-1-y: 0;  
  --bg-2-x: 0;
  --bg-2-y: 0;
  --bg-3-x: 0;
  --bg-3-y: 0; 
}

@supports (color: color(display-p3 1 1 1)) {
  :root {
    --color-1: color(display-p3 0.36 0.17 0.82);
    --color-2: color(display-p3 0.95 0.04 0.95);
    --color-3: color(display-p3 0.01 0.53 0.99);
  }
}

@media (min-width: 960px) {
  :root {
    --shadow-size: max(72px, 25vw);
    --shadow-size-fallback: 25vw;
    --shadow-blur: 80;
  }
}

* {
  box-sizing: border-box;
  outline: calc(var(--debug) * 1px) dashed red;
}

*::before,
*::after {
  outline: calc(var(--debug) * 1px) dashed red; 
}

html,
body {
  width: 100%;
  height: 125%;
  padding: 0;
  margin: 0;
}

body {
  font-family: "Mona Sans", sans-serif;
  display: grid;
  grid-template-columns: repeat(1, 1fr);
  background: var(--bg-color);
  z-index: 1;
  position: relative; 
}

body::before {
  content: "";
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  z-index: 1;
  background: radial-gradient(
        circle var(--shadow-size, var(--shadow-size-fallback)) at 20vw 0,
        var(--color-1, red) 100%,
        transparent 0
      ),
      radial-gradient(
        circle var(--shadow-size, var(--shadow-size-fallback)) at 100vw 0,
        var(--color-2, red) 100%,
        transparent 0
      ),
      radial-gradient(
        circle calc(var(--shadow-size, var(--shadow-size-fallback)) * 1.2) at
          50vw 115vh,
        var(--color-3, red) 100%,
        transparent 0
      );

  top: 0;
  left: 0;
  opacity: 0.5;
  filter: blur(calc(var(--shadow-blur) * 1px));
  mix-blend-mode: hue;
}

body::after {
  content: "";
  display: block;
  width: 100%;
  height: 100%;
  position: absolute;
  z-index: -1;
  top: 0;
  left: 0;
  filter: contrast(145%) brightness(650%) invert(100%);
  mix-blend-mode: screen;
  background: var(--bg-grain);
  background-size: 500px;
}

main {
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
  display: grid;
  place-items: center;
  z-index: 2;
}

main h1 {
  color: white;
  font-size: max(72px, 15vw);
  mix-blend-mode: lighten;
  font-weight: 650;
  font-stretch: 110%;
  letter-spacing: -0.04em;
  background: var(--bg-grain),
    conic-gradient(
      from 140deg at calc(var(--bg-1-x) * 1%) 90%,
      hsl(30deg 100% 5%),
      hsl(238deg 100% 5%),
      hsl(60deg 100% 99%),
      hsl(248deg 100% 31%),
      hsl(315deg 64% 51%),
      hsl(25deg 95% 61%),
      hsl(55deg 100% 75%),
      hsl(60deg 100% 99%),
      hsl(199deg 94% 74%),
      hsl(236deg 95% 28%),
      hsl(244deg 100% 4%)
    ),
    radial-gradient(
      ellipse at calc(var(--bg-2-x) * 1%) calc(var(--bg-2-y) * 1%),
      white 12%,
      transparent 35%
    ),
    radial-gradient(
      ellipse at calc(var(--bg-3-x) * 1%) calc(var(--bg-3-y) * 1%),
      hsl(212deg 94% 68%),
      transparent 35%
    );
 
    background: 
    var(--bg-grain),
    conic-gradient(/*...*/),
    radial-gradient(/*...*/),
    radial-gradient(/*...*/);
    
    background-repeat: repeat;
    background-size: 500px cover; /* Use 'cover' directly */
    background-blend-mode: color-burn;
    
    
    opacity: 1;
    animation: bg 20s linear infinite alternate;
    
}    
</style>
</head>
<body>
        <div>
        <div class="header">
            <h1 class="title">Chat With Any Document</h1>
            <p class="subtitle">Let's User get Summary from the document. </p>
          </div>
    </div>
    

    <form class="form">
        <label class="form__container" id="upload-container">Choose & Drop Document Files</label>
        <input class="form__file" id="upload-files" type="file" accept=".docx" multiple>
        <div id="files-list-container"></div>
      </form>

<div class="text-bar">
    <input class="words"   type="text"  placeholder="Type Your Text Here ">
</div>




<div id="processed-text" class="summary-bar">
    <input class="words"   type="text"  placeholder="Summary box ">
</div>
    
    







      <!-- <div class="major_text">
    <div style="display: flex; flex-direction: column; align-items: center" class="translate">
        <h2 style="margin-top: 48px; margin-bottom: 8px">NIRMAL THE BITCH ASS BOIIIII OF OUR CLASSS LESSS GOOO</h2>
        <span class="ant-typography ant-typography-secondary css-w8mnev" style="display: block; margin-bottom: 32px; font-size: 18px; text-align: center">Across borders, beyond languages: AI is revolutionizing the understanding of research worldwide</span>
    </div>
</div> -->

    <div class="features">
        <div class="feature">
             <h2><i class="fas fa-search"></i>For Students</h2> 
            <p>Prepare for exams, get help with homework and answer multiple choice questions.</p>
        </div>
        <div class="feature">
            <h2><i class="fas fa-flask"></i>For Researchers</h2>
            <p>Scientific papers, academic articles and books. Get the information you need for your research.</p>
        </div>
        <div class="feature">
            <h2><i class="fa-regular fa-notebook"></i>For Professionals</h2>
            <p>Legal contracts, financial reports, manuals and training material. Ask any question to any Document and get insights fast.</p>
        </div>
    </div>
    </div>





    <script>
        const multipleEvents = (element, eventNames, listener) => {
            const events = eventNames.split(' ');

            events.forEach(event => {
                element.addEventListener(event, listener, false);
            });
        };

        const fileUpload = () => {
            const INPUT_FILE = document.querySelector('#upload-files');
            const INPUT_CONTAINER = document.querySelector('#upload-container');
            const PROCESSED_TEXT_DIV = document.querySelector('#processed-text');

            const showProcessedText = (processedText) => {
                // Display the processed text in the designated div
                PROCESSED_TEXT_DIV.textContent = processedText;
            };

            multipleEvents(INPUT_FILE, 'click dragstart dragover', () => {
                INPUT_CONTAINER.classList.add('active');
            });

            multipleEvents(INPUT_FILE, 'dragleave dragend drop change', () => {
                INPUT_CONTAINER.classList.remove('active');
            });

            INPUT_FILE.addEventListener('change', () => {
                const files = [...INPUT_FILE.files];

                files.forEach(file => {
                    if (file.type === 'application/msword' || file.type ===
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
                        const reader = new FileReader();

                        reader.onload = function (event) {
                            const arrayBuffer = event.target.result;
                            const formData = new FormData();
                            formData.append('file', new Blob([arrayBuffer]));

                            // Send the file to the Flask server for text extraction and processing
                            fetch('http://127.0.0.1:5000/process-text', {
                                    method: 'POST',
                                    body: formData
                                })
                                .then(response => {
                                    // Check if the response status is OK (200) or not
                                    if (response.ok) {
                                        // Parse the JSON response
                                        return response.json();
                                    } else {
                                        // If response status is not OK, handle it accordingly (e.g., throw an error)
                                        throw new Error('Network response was not ok: ' + response.statusText);
                                    }
                                })
                                .then(data => {
                                    // Handle the JSON data here
                                    const processedText = data.processedText;

                                    // Call the function to display processed text
                                    showProcessedText(processedText);
                                })
                                .catch(error => {
                                    // Handle any errors that occurred during the fetch operation
                                    console.error('Error:', error);
                                });
                        };

                        reader.readAsArrayBuffer(file);
                    } else {
                        // Handle other file types if needed
                        console.log('Unsupported file type:', file.type);
                    }
                });
            });
        };

        fileUpload();
    </script>


    </body>
    </html>
    """,width=1000,height=1000 
)



# HtmlFile = open("index.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# print(source_code)
# components.html(source_code)

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

REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request Latency',
                           ['method', 'endpoint'])
REQUEST_COUNT = Counter('request_count', 'Request Count',
                        ['method', 'endpoint', 'http_status'])
ERROR_COUNT = Counter('error_count', 'Error Count',
                      ['method', 'endpoint'])

@app.route('/metrics', methods=['GET'])
def metrics():
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)

