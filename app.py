from flask import Flask, escape, request
from definition_extractor import multi_definition_extraction

app = Flask(__name__)

@app.route('/glossary_builder/')
def hello():
    sentence = request.args.get("sentence", "Photosynthesis is the process of manufacturing food using sunlight.")
    # return f'Hello, {escape(name)}!'
    return multi_definition_extraction(sentence).values

@app.route('/')
def welcome():
    return "<h1>Welcome to our project!</h1>"

if __name__== '__main__':
    app.run(host="0.0.0.0", port=8000)
