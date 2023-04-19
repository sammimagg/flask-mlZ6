from flask import Flask, request, jsonify
from transformers import pipeline, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import os
from flask_cors import CORS
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

config_fake_news = AutoConfig.from_pretrained('./models/fake_news_model')
tokenizer_fake_news = AutoTokenizer.from_pretrained('./models/fake_news_model')
model_fake_news = AutoModelForSequenceClassification.from_pretrained('./models/fake_news_model', config=config_fake_news)
classifier_fake_news = pipeline('text-classification', model=model_fake_news, tokenizer=tokenizer_fake_news)

config_job_scams = AutoConfig.from_pretrained('./models/job_scams_model')
tokenizer_job_scams = AutoTokenizer.from_pretrained('./models/job_scams_model')
model_job_scams = AutoModelForSequenceClassification.from_pretrained('./models/job_scams_model', config=config_job_scams)
classifier_job_scams = pipeline('text-classification', model=model_job_scams, tokenizer=tokenizer_job_scams)

config_phishing = AutoConfig.from_pretrained('./models/phishing_model')
tokenizer_phishing_model = AutoTokenizer.from_pretrained('./models/phishing_model')
model_phishing_model = AutoModelForSequenceClassification.from_pretrained('./models/phishing_model', config=config_phishing)
classifier_phishing_model = pipeline('text-classification', model=model_phishing_model, tokenizer=tokenizer_phishing_model)

config_political_statements = AutoConfig.from_pretrained('./models/political_statements_model')
tokenizer_political_statements = AutoTokenizer.from_pretrained('./models/political_statements_model')
model_political_statements = AutoModelForSequenceClassification.from_pretrained('./models/political_statements_model', config=config_political_statements)
classifier_political_statements = pipeline('text-classification', model=model_political_statements, tokenizer=tokenizer_political_statements)

config_product_reviews = AutoConfig.from_pretrained('./models/product_reviews_model')
tokenizer_product_reviews = AutoTokenizer.from_pretrained('./models/product_reviews_model')
model_product_reviews = AutoModelForSequenceClassification.from_pretrained('./models/product_reviews_model', config=config_product_reviews)
classifier_product_reviews = pipeline('text-classification', model=model_product_reviews, tokenizer=tokenizer_product_reviews)

app = Flask(__name__)
CORS(app)


@app.route('/predict-fake-news', methods=['POST'])
def predict_fake_news():
    text = request.json['text']
    result = classifier_fake_news(text)
    label = True if result[0]['label'] == "LABEL_1" else False
    score = result[0]['score']
    response = {'is_deceptive': label, 'score': score}
    return jsonify(response)

@app.route('/predict-job-scams', methods=['POST'])
def predict_job_scams():
    text = request.json['text']
    result = classifier_job_scams(text)
    label = True if result[0]['label'] == "LABEL_1" else False
    score = result[0]['score']
    response = {'is_deceptive': label, 'score': score}
    return jsonify(response)

@app.route('/predict-phishing', methods=['POST'])
def predict_phishing():
    text = request.json['text']
    result = classifier_phishing_model(text)
    label = True if result[0]['label'] == "LABEL_1" else False
    score = result[0]['score']
    response = {'is_deceptive': label, 'score': score}
    return jsonify(response)

@app.route('/predict-political-statements', methods=['POST'])
def predict_political_statements():
    text = request.json['text']
    result = classifier_political_statements(text)
    label = True if result[0]['label'] == "LABEL_1" else False
    score = result[0]['score']
    response = {'is_deceptive': label, 'score': score}
    return jsonify(response)

@app.route('/predict-product-reviews', methods=['POST'])
def predict_product_reviews():
    text = request.json['text']
    result = classifier_product_reviews(text)
    label = True if result[0]['label'] == "LABEL_1" else False
    score = result[0]['score']
    response = {'is_deceptive': label, 'score': score}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
