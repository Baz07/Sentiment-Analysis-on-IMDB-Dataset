## Import Required Libraries
import random
import tensorflow
import os ## For Reading Data Path purpose
import tensorflow
import numpy as np
import pandas as pd
import string ## To remove all single alphabets from Reviews
from nltk.corpus import stopwords ## TO Remove Stopwords from Reviews
from string import punctuation ## To remove punctuations from reviews
from bs4 import BeautifulSoup ## To Remove HTML Tags fro reviews
import re ## To Remove unwanted characters and words from reviews
from nltk.tokenize import TweetTokenizer ## To Create Tokens
from keras.preprocessing.text import Tokenizer ## For Text to Vector(Sequence of Integers) Conversion
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense, Conv1D ## Layers for Neural Network
from keras import Sequential ## For Creating a Sequential Model of Neural Network
import pickle ## TO Load the Saved Trained Model
from sklearn.metrics import accuracy_score ## To Measure accuracy of predicted labels and original labels
from keras.models import load_model ## For loadinf the Trained Model
import nltk ## For Preprocessing purposes
nltk.download('stopwords') ## To Download Stowords library


## Function to load the testing data and testing labels
def load_test_data(data_path):

    ## Set Seed
    np.random.seed(1551)
    tensorflow.random.set_seed(1551)

    ## Provide Dataset folder path 
    dataset_path= os.path.join(data_path, 'aclImdb')

    ## List to store Testing Data and Labels (Following same procedure as loading Training Data)
    Data = []
    Label = []
    sub_folder = ['pos', 'neg']
    for each_folder in sub_folder:
        Inside_train_folder = os.path.join(dataset_path, 'test', each_folder) ## Join provided path with "Test" subfolder
        All_files = os.listdir(Inside_train_folder) # List all directories of "test" folder
        Sorted_files = sorted(All_files) ## Sort the Files
        for file in Sorted_files: ## Loop for "pos" and "neg" file in order to capture the reviews
            if file.endswith('.txt'): ## Check only for Text Files
              file_path = os.path.join(Inside_train_folder, file) ## Selecting the Text File
              with open(file_path, encoding='utf-8') as required_file: ## Open File
                Data.append(required_file.read()) ## Append the data in "Data"
              
              ## Store Review Sentiment Polarity Scores in Label list
              if each_folder == 'neg': ## If Folder is "neg", i.e. all the reviews are negative and therefore Label would be 0, otherwise for "pos" i.e. positive reviews, Label will be 1.
                Label.append(0)
              else:
                Label.append(1)
    return (Data, Label)  ## Return a tuple which contains "Complete Testing Data" and "Complete Testing Labels"

if __name__ == "__main__":
  ## 1. Load the Trained Model
  Model = load_model("models/NLP_model.h5")

  ## Load the Tokenizer from "train_NLP" File (Needed to perform "Texts to Sequence" and "Pad Sequences" Operation)
  Text_Sequence_Convertor = pickle.load(open("models/Sequence_convertor.pkl", "rb"))

  ## 2. Load Testing Data
  Testing_data = load_test_data("data")

  ## Segregate Reviews and their polarities/labels from Testing Data
  Test_data = Testing_data[0]
  Test_label = Testing_data[1]
  # print("Testing Data length: ", len(Test_data))
  # print("Testing_label_length: ", len(Test_label))

  ## Convert to Training Review Data and Label DataFrame for further processing
  Testing_data = pd.DataFrame(Test_data)
  Testing_label = pd.DataFrame(Test_label)
  # print("Testing Data:\n", Testing_data)
  # print('---------------------------------------')
  # print("Testing Label:\n", Testing_label)


  ## Creating a list of all single letter alphabets (Will be removed since they are meaningless)
  Alphabets = [] ## List to store all alphabets
  string = string.ascii_letters ## Store all ASCII letters (string: 'abcdefghijklmnopqrstuvwxyz')
  for i in string:
    alphabet = i.split() ## Need to split each alphabet in order to save it as an individual alphabet in the list
    alphabet = "".join(alphabet) ## Convert each individual alphabet into string
    # print(type(alphabet))
    Alphabets.append(alphabet) ## Append each string alphabet into "Alphabets" List


  ## Custom Stopwords List (Created Manually)
  stopwords = ['all', 'few', "i'll",'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'any', 'both', 'each', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma','shan']

  ## Create a list of all punctuations (which will be removed later on from all the reviews)y
  punc = list(punctuation)

  ## Combining Stopwords list, single alphabets list and punctuation symbols list to create a single list
  Custom_words= [*stopwords, *Alphabets, *punc]
  # print("All Custom Words: ", Custom_words)
  # print("Total Number of Custom Words: ", len(Custom_words))

  ## Convert Testing Data into Numpy Array for further processing
  Testing_text = Testing_data.values
  # print("Example of Testing Review:\n", Testing_text[50])
  # print('\n-----------------------------------\n')

  ## Removing HTML tags present in Reviews using Beautiful Soup Library HTML Parser and other noise present in data using regular expressions.
  Clean_Testing_text = [] ## List to store clean Testing Reviews
  for review in Testing_text:
    parsed_review = BeautifulSoup(str(review), "html.parser") ## Create a HTML Parser
    clean_review = parsed_review.get_text() ## Force HTML Parser on each review which will remove all HTML Tags present in reviews
    clean_review = re.sub(r'\.+',"",clean_review) ## Remove Unwanted Symbols such as "....."
    # clean_review = re.sub(r'[0-9]+',"",clean_review)
    Clean_Testing_text.append(clean_review)
  # print("Example of Clean Testing Review:\n", Clean_Testing_text[50])


  ## Implement Tweet Tokenizer to obtain Tokens(each word separately) from each review.
  Test_tokens= [TweetTokenizer().tokenize(sentence) for sentence in Clean_Testing_text]
  # print("Tokens of a Testing Review:\n", Test_tokens[0])

  ## Custom Word Removal Process and Lowercase all the tokens present in the reviews.
  test_token_list = [] ## List to store tokens of all testing reviews
  for i in Test_tokens:
    each_token_list = []
    for j in i:
      j = j.lower() ## Lowercase all words before checking if it is a stopword or not.
      if j not in Custom_words: ## Check if token is not present in Custom Word List, add it to "train_token_list", i.e. all required and meaningful tokens
        each_token_list.append(j)
    test_token_list.append(each_token_list)
  # print('\n-------------------------------------------------\n')
  # print("Tokens of Clean Testing Review:\n", test_token_list[0])

  ## Concatenate the tokens of each review to a single string
  Test_Reviews_with_main_words = []
  for i in test_token_list:
    review = " ".join(i)
    # print(review)
    # print('-----------------------')
    Test_Reviews_with_main_words.append(review)
  # print("Example of Testing Review after combining all its token:\n", Test_Reviews_with_main_words[0])

  ## Converting Testing reviews to a sequence vector
  Test_review_sequences = Text_Sequence_Convertor.texts_to_sequences(Test_Reviews_with_main_words)

  ## Creating a pad sequence to make sure that the length of all testing reviews sequences should be same.
  padded_testing_reviews = pad_sequences(Test_review_sequences, truncating = 'post', maxlen = 1000)
  # print("Result after padding first testing review:\n", len(padded_testing_reviews[0]))
  # print("Result after padding second testing review:\n", len(padded_testing_reviews[1]))

  ## Converting Testing Labels into list
  ts_labels = []
  for each_label in Testing_label[0]:
    # print(each_label)
    ts_labels.append(each_label)

  ## Convert Testing Label List to Numpy array
  testing_reviews_label = np.asarray(ts_labels)
  # print("Testing Review Labels:\n", testing_reviews_label)

  ## 3. Prediction on Padded Testing Reviews
  Testing_SA = Model.predict(padded_testing_reviews) ## Testing_SA: Testing Model of Sentiment Analysis

  ## Creating Final Labels from Predicted Labels
  Final_labels = [] ## List to store final labels
  for value in Testing_SA:
    '''
    New Labels will be on the basis of following condition:
    - If Review Polarity is greater than 0.5, it will be a positive review and its value will be 1
    - If Review Polarity is less than 0.5, it will be a negative review and its value will be 0

    '''
    if value > 0.5:   ## Positive Label will be assigned 0
      Final_labels.append(1)
    else: ## Negative Label will be assigned 1
      Final_labels.append(0)

  ## Converting Final labels to Numpy array in order to match the datatype of both the inputs to "accuracy_score"
  Predicted_Testing_label = np.asarray(Final_labels)
  print("Predicted testing labels:\n", Predicted_Testing_label)

  ## Calculate Accuracy by comapring Prediction and Original Test Labels
  Acc = accuracy_score(Predicted_Testing_label, testing_reviews_label)
  print("Final Accuracy on Testing Data: ", Acc*100, "%")

## Final Accuracy Achieved on Testing Data:  86.9 %
