# Political Sentiment Analysis

This is a library to analyze the political sentiment of text. We use Deep Learning models to evaluate the speaker's political leaning (liberal/democrat or conservative/republican). This was done as the final project to an MIT course, 17.835 in 2018. Trained models can be found in the bin.

# Setup

After cloning this repository, setup a python virtual environment. This can be done with 

```
source setup
```

this can also be used after initially setting up the environment to activate the virtual environment.

# Project Structure

The project is structured in the following fashion. 

```
.
+-- _setup
+-- _bin
|   +-- this includes binary files like weights 
|       or cached data files. 
+-- _data
|   +--_convote
|       +-- this is the cornell vote database 
|           that includes transcriptions of congressional
|           debates from 2005 along with labels for 
|           both for the political party of the speaker and 
|           whether the statement is for/against the bill.
|   +--_twitter
|       +-- data we have collected from the public twitter api.
+-- _preprocessing 
|        +-- code to preprocess the various datasets into usable formats 
|	    for the various models 
+-- _models
|   +-- contains python code that exports keras models

```

