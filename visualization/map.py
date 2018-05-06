# system
import math

# lib
import plotly.plotly as py
import plotly

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np

# self
from data.twitter.data_collection import TwitterClient
from data.twitter.data import get_political_words
from models.recurrent import RecurrentSentimentModel
from preprocessing.preprocess import clean_text_documents
from keras.preprocessing.sequence import pad_sequences

GEOBOX_WORLD = [-180,-90,180,90]

app = dash.Dash()
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})
model = None

long = []
lat = []
label = []
tweet = []

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


app.layout = html.Div(children=[
    html.Div(className="container", children=[
        html.H1(children='Twitter Live Political Sentiment Classification'),
        html.Div(children='''Dash: A web application framework for Python.'''),

        dcc.Graph(id='graph', animate=True),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        ),
    ]),
])

@app.callback(Output('graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def gen_plot(n):

    layout = dict(
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.8,
            subunitwidth=0.8
        ),
    )

    data = [dict(
        type='scattergeo',
        locationmode='USA-states',
        lon=long,
        lat=lat,
        text=tweet,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            symbol='square',
            line=dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            cmin=0,
            color=label,
        ))]

    fig = dict(data=data, layout=layout)

    fig['layout']['margin'] = {
        'l': 0, 'r': 0, 'b': 0, 't': 0
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    return fig


def applier(status):
    if status.place is not None:
        x, y = status.place.bounding_box.coordinates[0][0][0], \
               status.place.bounding_box.coordinates[0][0][1]

        print(x,y)

        samples = clean_text_documents([status.text])
        tweet.append(status.text)

        formatted = [[model.word_to_index[w] if w in model.word_to_index else model.word_to_index['oov']
                      for w in sample.split()] for sample in samples]
        formatted = np.array(formatted)
        formatted = pad_sequences(formatted, maxlen=model.input_length)

        result = model.model.predict(formatted)[0][0]

        long.append(x)
        lat.append(y)
        label.append(result)


def stream():
    client = TwitterClient(stream_func=applier)
    client.stream.filter(track=political_words, async=True)


if __name__ == '__main__':
    load_model()
    stream()
    app.run_server(debug=True)
