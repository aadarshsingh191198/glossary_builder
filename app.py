from flask import Flask, escape, request
from definition_extractor import definition_extraction

app = Flask(__name__)

@app.route('/glossary_builder/')
def hello():
    sentence = request.args.get("sentence", "This is a simple definition.")
    # return f'Hello, {escape(name)}!'
    return definition_extraction(sentence)

if __name__== '__main__':
    app.run()