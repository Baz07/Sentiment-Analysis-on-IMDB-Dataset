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
import nltk ## For Preprocessing purposes
import pickle ## To save Tokenizer  
nltk.download('stopwords') ## To Download Stowords library

## Set Seed to produce reproducible results
np.random.seed(123)
tensorflow.random.set_seed(123)

## Function to load the training data and training labels
def load_train_data(data_folder_path):

  ## Provide Dataset folder path 
  dataset_path= os.path.join(data_folder_path, 'aclImdb')

  ## Define Data and Label list to store Review Data and their corresponsding sentiment polarities (labels).
  Data = []
  Label = []

  ## Define a list containing the name of sub folders present in data folder for navigation.
  sub_folder = ['pos', 'neg']

  ## Read and Store Positive and Negative reviews in a single list named "Data"
  for each_folder in sub_folder:
      Inside_train_folder = os.path.join(dataset_path, 'train', each_folder) ## Join provided path with "Train" subfolder 
      All_files = os.listdir(Inside_train_folder) # List all directories of "train" folder
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

  return (Data, Label)  ## Return a tuple which contains "Complete Training Data" and "Complete Training Labels"

if __name__ == "__main__":

  ## 1. Load Train Data using above mentioned function
  Training_data = load_train_data("data")
  
  ## 2. Network Training
  ## Segregate Training Reviews and their polarities/labels from Training Data 
  Train_data = Training_data[0]
  Train_label = Training_data[1]
  # print("Training Data length: ", len(Train_data))
  # print("Training_label_length: ", len(Train_label))

  ## Convert Training Data and Labels into dataframe
  Training_data = pd.DataFrame(Train_data)
  Training_label = pd.DataFrame(Train_label)
  # print("Training Data:\n", Training_data)
  # print('---------------------------------------')
  # print("Training Label:\n", Training_label)
  # print('---------------------------------------')


  ## Import and Download NLTK Stopwords

  '''
  - Stopwords are unwanted words which does not help in producing any significant result of sentiment analysis.
  - It is considered a good practice to remove stopwords, however NLTK Stopwords library contains some words (Ex: Don't , shouldn't, etc.) which shouldn't be omitted from the data
    (Reason being they change the context of the data)
    
  ## Required Stopwords list (few custom words are also added):
  ['all', 'few', "i'll", 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma','shan', 'no']

  '''

  ## Creating a list of all single letter alphabets (Will be removed since they are meaningless)
  Alphabets = [] ## List to store all alphabets
  string = string.ascii_letters ## Store all ASCII letters (string: 'abcdefghijklmnopqrstuvwxyz')
  for i in string:
    alphabet = i.split()  ## Need to split each alphabet in order to save it as an individual alphabet in the list
    alphabet = "".join(alphabet) ## Convert each individual alphabet into string
    # print(type(alphabet))
    Alphabets.append(alphabet) ## Append each string alphabet into "Alphabets" List

  ## Custom Stopwords List (Created Manually)
  stopwords = ['all', 'few', "i'll",'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'any', 'both', 'each', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ma','shan']

  ## Create a list of all punctuations (which will be removed later on from all the reviews)
  punc = list(punctuation)

  ## Combining Stopwords list, single alphabets list and punctuation symbols list to create a single list
  Custom_words= [*stopwords, *Alphabets, *punc]
  # print("All Custom Words: ", Custom_words)
  # print("Total Number of Custom Words: ", len(Custom_words))

  ## Converting Training Data to Array for further processing
  Training_text = Training_data.values
  # print("Original Tweet:\n", Training_text[0])
  # print('\n-----------------------------------\n')

  ## Removing HTML tags present in Reviews using Beautiful Soup Library HTML Parser and other noise present in data using regular expressions.
  Clean_Training_text = [] ## List to Store Clean Training Reviews
  for review in Training_text:
    parsed_review = BeautifulSoup(str(review), "html.parser")  ## Create a HTML Parser
    clean_review = parsed_review.get_text() ## Force HTML Parser on each review which will remove all HTML Tags present in reviews
    clean_review = re.sub(r'\.+',"",clean_review) ## Remove Unwanted Symbols such as "....." using Regex
    # clean_review = re.sub(r'[0-9]+',"",clean_review)
    Clean_Training_text.append(clean_review)

  # print("Cleaned Tweet:\n", Clean_Training_text[0])

  ## **NOTE:** Here I would like to mention one important thing is I chose **not to remove** Numbers from Training or Testing Data as they provide important metric while classifying tweets as positive or Negative."""

  ## Implement Tweet Tokenizer to obtain Tokens(each word separately) from each review.
  Train_tokens= [TweetTokenizer().tokenize(sentence) for sentence in Clean_Training_text]
  # print("Tokens of Review:\n", Train_tokens[0])

  ## Custom Word Removal Process and Lowercase all the tokens present in the reviews.
  train_token_list = []  ## List to store tokens of all training reviews
  for i in Train_tokens:
    each_token_list = []
    for j in i:
      j = j.lower()  ## Lowercase all words before checking if it is a stopword or not.
      if j not in Custom_words: ## Check if token is not present in Custom Word List, add it to "train_token_list", i.e. all required and meaningful tokens
        each_token_list.append(j)
    train_token_list.append(each_token_list)
  # print('\n-------------------------------------------------\n')
  # print("Tokens of a Review after removing Custom Stopwords:\n", train_token_list[0])

  ## Concatenate the tokens of each review to a single string
  Reviews_with_main_words = []
  for i in train_token_list:
    review = " ".join(i)
    # print(review)
    # print('-----------------------')
    Reviews_with_main_words.append(review)
  # print(Reviews_with_main_words[0])


  ## Create Tokenizer to convert Reviews to their specific vectors
  Text_Sequence_Convertor = Tokenizer(num_words= 10000,oov_token = '<OOV>') ## "OOV Token" has been used to replace "Out of Vocabulary" Words during the text to sequence conversion

  ## Fit Convertor on Reviews that contains only meaningful words
  Text_Sequence_Convertor.fit_on_texts(Reviews_with_main_words)

  ## Assign corresponding integer to the words present in Reviews (Values will be assigned according to data which was fitted earlier in Convertor)
  Training_review_sequence = Text_Sequence_Convertor.texts_to_sequences(Reviews_with_main_words)
  # print("Generated Sequence of a Training Review:\n", Training_review_sequence[0])

  ## Capture the Total Vocabulary present in Training Reviews
  word_index = Text_Sequence_Convertor.word_index
  # print("Total Vocabulary of Training reviews: ", len(word_index))

  ## Saving Tokenizer to be used in Testing File

  pickle.dump(Text_Sequence_Convertor, open("models/Sequence_convertor.pkl","wb"))

  ## Creating a pad sequence to make sure that the length of all reviews sequences should be same.
  '''
  Pad Sequences Parameters:
  - maxlen: 1000 --> Each review will have maximum length of 1000
  - Truncating: POST --> End Part of review will be truncated if length exceeds.

  '''
  padded_training_reviews = pad_sequences(Training_review_sequence, truncating = 'post', maxlen = 1000)

  # print("Result after padding first training review:\n", len(padded_training_reviews[0]))
  # print("Result after padding second training review:\n", len(padded_training_reviews[1]))

  ## Converting Training Labels to list
  tr_labels = [] ## List to store Training Labels
  for each_label in Training_label[0]:
    # print(each_label)
    tr_labels.append(each_label)

  ## Converting Training Label List to Training Numpy Array
  training_reviews_label = np.asarray(tr_labels)
  # print("Training Review Labels:\n", training_reviews_label)

  ##************************************************NN MODEL CONSTRUCTION***************************************************
  ## Design the Neural Network for Training
  ## Embedding Layer Parameters
  vocab_size = 112594 
  emb_dimension = 18
  review_max_length = 1000

  ## Dense Layer Parameters
  D_neurons = 6

  ## Output Layer Neuron
  O_neuron  = 1

  Model = Sequential([
              Embedding(vocab_size, emb_dimension, input_length = review_max_length),
              Flatten(),
              Dense(D_neurons, activation = 'relu'),
              Dense(O_neuron, activation = 'sigmoid'),
  ])

  ## Compile Designed Model with Loss, Optimizer and Metrics
  Model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

  ## Print Model Summary
  Model.summary()

  ## Fit Model with Training Reviews
  print("START TRAINING:\n")
  Training_SA = Model.fit(padded_training_reviews, training_reviews_label, epochs = 10) ## Training_SA = Training Sentiment Analysis Model
  print("END TRAINING\n")

  ## Evaluate Model
  Score = Model.evaluate(padded_training_reviews, training_reviews_label)
  print('-----------------------------------------------------------------------')
  print("Final Training Loss: ", Score[0])
  print("Final Training Accuracy: ", Score[1]*100)
  print('-----------------------------------------------------------------------')

  ## 3. Save Model
  Model.save("models/20842555_NLP_model.h5")
