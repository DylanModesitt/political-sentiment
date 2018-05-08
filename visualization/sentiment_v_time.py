import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
import numpy as np

# self
from data.twitter.data_collection import TwitterClient
from data.twitter.data import get_political_words
from models.recurrent import RecurrentSentimentModel
from preprocessing.preprocess import clean_text_documents
from keras.preprocessing.sequence import pad_sequences

new_raw_data = []
raw_data = []
data = {}
count = {}

with open('./data/twitter/meta/political_words_verified.txt') as f:
    political_words = f.read().splitlines()

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

#popular topics: google, olympics, trump, gun, usa

app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.Div(className="container", children=[
        html.H1(children="Twitter Live Political Sentiment Classification"),
        html.Div(children='''Dash: A web application framework for Python.'''),

        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*3000
        ),
    ]),
])

@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    global new_raw_data
    global raw_data
    global data
    global count

    try:
        for date, label in new_raw_data:
            if date not in data:
                data[date] = label
                count[date] = 1
            else: # Update average
                data[date] = (data[date] * (count[date] / (count[date] + 1))) + (label / (count[date] + 1))
                count[date] += 1

        raw_data += new_raw_data
        new_raw_data = []

        data_list = list(data.items())

        data_list.sort(reverse=True)
        X, Y = zip(*data_list)
        X = X[:1000]
        Y = Y[:1000]

        plot = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [plot],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),)}

    except Exception as e:
        print(str(e))
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

def sql_applier(status):
    samples = clean_text_documents([status.text])

    formatted = [[model.word_to_index[w] if w in model.word_to_index else model.word_to_index['oov']
                      for w in sample.split()] for sample in samples]
    formatted = np.array(formatted)
    formatted = pad_sequences(formatted, maxlen=model.input_length)

    result = model.model.predict(formatted)[0][0]
    new_raw_data.append((status.created_at.strftime("%Y-%m-%d %H:%M:%S"), result))

def stream():
    client = TwitterClient(stream_func=sql_applier)
    client.stream.filter(track=political_words, async=True)

if __name__ == '__main__':
    load_model()
    stream()
    app.run_server(debug=True, port=8000)
