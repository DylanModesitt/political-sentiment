# system
import json

# self
import numpy as np
from models.recurrent import RecurrentSentimentModel
from preprocessing.preprocess import clean_text_documents, process_data
from keras.preprocessing.sequence import pad_sequences


if __name__ == '__main__':

    model = RecurrentSentimentModel(dir='./bin/main')
    model.load()

    with open('./data/twitter/nkdumas.json') as f:
        data = json.load(f)
        tweets = [t['content'] for t in data[0]['tweets']]

    samples = clean_text_documents(tweets, twitter=True)
    formatted = [[model.word_to_index[w] if w in model.word_to_index else model.word_to_index['oov']
                  for w in sample.split()] for sample in samples]

    formatted = pad_sequences(formatted, maxlen=model.input_length)

    # model.model.fit(x=formatted,
    #                 y=np.array([0]*len(formatted)),
    #                 epochs=1,
    #                 batch_size=32)

    p = model.model.predict(formatted).flatten()

    print(np.mean(p))
    pred = list([float(e) for e in model.model.predict(formatted).flatten()])

    together = list(zip(samples, pred))
    with open('./the.txt', 'w') as f:
        json.dump(together, f, indent=4)



