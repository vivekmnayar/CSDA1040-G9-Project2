#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

grey_button_style = {'background-color': 'grey',
                      'color': 'black'}

app.layout = html.Div([
    html.Div([
        html.Br(),
    html.Button('Generate Story', id='btn-nclicks-1', n_clicks=0, style=grey_button_style
        )
    ], style={'width':'95%', 'margin':25, 'textAlign': 'center'}, className="twelve columns"),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div(id='container-button-timestamp')
    ], style={'width':'95%', 'margin':25, 'textAlign': 'center'}, className="twelve columns")
], style={'backgroundColor':'lightblue'})

@app.callback(Output('container-button-timestamp', 'children'),
              [Input('btn-nclicks-1', 'n_clicks')])
def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        filename = "wonderland.txt"
        raw_text = open(filename, 'r', encoding='utf-8').read()
        raw_text = raw_text.lower()
        # create mapping of unique chars to integers, and a reverse mapping
        chars = sorted(list(set(raw_text)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))
        # summarize the loaded data
        n_chars = len(raw_text)
        n_vocab = len(chars)
        #print ("Total Characters: ", n_chars)
        #print ("Total Vocab: ", n_vocab)
        # prepare the dataset of input to output pairs encoded as integers
        seq_length = 100
        dataX = []
        dataY = []
        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
        n_patterns = len(dataX)
        #print ("Total Patterns: ", n_patterns)
        # reshape X to be [samples, time steps, features]
        X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
        # normalize
        X = X / float(n_vocab)
        # one hot encode the output variable
        y = np_utils.to_categorical(dataY)
        # define the LSTM model
        model = Sequential()
        model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        # load the network weights
        filename = "weights-improvement-50-1.2028-bigger.hdf5"
        model.load_weights(filename)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        result2 = ''
        # pick a random seed
        start = numpy.random.randint(0, len(dataX)-1)
        pattern = dataX[start]
        #print ("Seed:")
        print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(1000):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            result2 += result
        #print ("\nDone.")
        msg = result2
    else:
        msg = 'Click the button to read a story.'
    return html.Div(msg)

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




