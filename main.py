from flask import Flask, request, jsonify
from transformers import pipeline, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import os
#from flask_cors import CORS
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

config = AutoConfig.from_pretrained('./model')
tokenizer = AutoTokenizer.from_pretrained('./model')
model = AutoModelForSequenceClassification.from_pretrained('./model', config=config)

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

app = Flask(__name__)
#CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']
    response = {'label': label, 'score': score}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
