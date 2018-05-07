# system
import io

# import
import numpy as np
import flask
from flask import jsonify
from keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

# self
from models.recurrent import RecurrentSentimentModel
from preprocessing.preprocess import clean_text_documents

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)

model = None


def load_model():
    """
    load the political sentiment deep learning model
    being used and set it as a global.

    :return: the model
    """
    global model

    model = RecurrentSentimentModel(dir='./bin/main')
    model.load()
    model.model.predict(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                  ,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                  ,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                  ,3,1,501,40,375,15,1999,4]]))

    print('model loaded')


load_model()


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    """
    predict the political sentiment of a given
    sentence.

    *** POST
    this route expects post data of the format

    {
        'text': 'lorem ipsum sic dolar...'
    }

    and will return a label between 0 and 1,
    0 representing liberal and 1 representing
    concervative

    :return:
    """

    text = flask.request.values.get('text')
    samples = clean_text_documents([text])

    formatted = [[model.word_to_index[w] if w in model.word_to_index else
                  model.word_to_index['oov'] for w in sample.split()] for sample in samples]
    formatted = np.array(formatted)
    formatted = pad_sequences(formatted, maxlen=model.input_length)

    result = model.model.predict(formatted)
    result = result[0][0]

    label = [1 - result, result]

    response = jsonify({
        'label': [str(r) for r in label],
        'class_names': ['liberal', 'conservative']
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    # load_model()
    app.run()
