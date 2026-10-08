# step1 = impport required libraries

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

####################################################################################
# step 2 = configuration of values
####################################################################################

VOCAB_SIZE = 10000  # consider most frequent 10000 unique words
MAX_LENGTH = 200  #consider maximum 200 words in review

####################################################################################
# step 3 = Load the IMDB dataset
####################################################################################

print("-"*40)
print("Movie sentiment analysis using LSTM")
print("-"*40)

print("Loading the dataset")

(X_train , Y_train) , (X_test ,Y_test) = imdb.load_data(num_words = VOCAB_SIZE)

print("IMDB dataset loaded successfully")

print("NUmber of training reviews :", len(X_train))
print("Number of testing reviews ", len(X_test))

####################################################################################
#  X_train= reviews used for training 
#  Y_train =actual sentiment of training 
#  X_test = reviews used for testing
#  Y_test = actual sentiment for testing

# sentiment :
# 0 -> Negative sentiment
# 1 -> positive sentiment
####################################################################################

####################################################################################
# step 4 = Load the word dictonary
####################################################################################

word_index = imdb.get_word_index()

#dictonary contains mapping of word and its corresponding number
# Drishyam is good movie     -> (20 56 78 43)
# 20 -> drishyam
# 56 -> is
# 78 -> good
# 43 -> movie


####################################################################################
# step 5 = create reverse dictonary
####################################################################################

reverse_word_index = {}

for word , index in word_index.items():
    reverse_word_index[index+3] = word
    


