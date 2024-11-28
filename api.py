from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import sys
from model import *
from config import *

app = Flask(__name__)
# @app.route('/')
CORS(app)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/api/question', methods=['POST'])
def post_question():
    json = request.get_json(silent=True)
    question = json['question']
    logging.info("Question received: %s", question)

    resp = chat(question)
    data = {'answer': resp}

    return jsonify(data), 200

if __name__ == '__main__':
    init_llm()
    index = init_index()
    init_query_engine(index)

    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
