# step1 = impport required libraries

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

####################################################################################
# step 2 = configuration of values
####################################################################################

VOCAB_SIZE = 10000  # consider most frequent 10000 unique words
MAX_LENTH = 200  #consider maximum 200 words in review

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
    
####################################################################################
# step 6  = function to decode review  (number to word)
####################################################################################

def DecodeReview(encode_review):
    words = []
    
    for number in encode_review:
        if number >= 3 : #ignore first 3
            word = reverse_word_index.get(number,"?")
            words.append(word)
            
    return "".join(words)  # join the list of the words

####################################################################################
# step 7  = Display sample reviews
####################################################################################

print("-"*40)
print("---------------------sample reviews-----------------------")
print("-"*40)

for i in range(3,7):
    review = DecodeReview(X_train[i])
    
    print("-"*40)

    print("review number", i+1)
    print("review:")
    print(review)
    
    print("-"*40)
    
    if Y_train[i]==1:
        print("sentiment; positive")
    else:
        print("sentiment : negative")
        
####################################################################################
# step 8  = padding
####################################################################################
   
X_train_padded = pad_sequences(
    X_train ,
    maxlen = MAX_LENTH 
)
    
X_test_padded = pad_sequences(
    X_test ,
    maxlen = MAX_LENTH
)

print("training data shape: ",X_train_padded.shape)
print("testing data shape: ", X_test_padded.shape)

####################################################################################
# step 9  = create LSTM model
####################################################################################
        
model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=32             #each word is represented in 32 values
    )
)

model.add(
    LSTM(
       units = 64                # size of LSTM hidden state 
    )
    
)
model.add(
    Dense(
        units = 1,                  # one output
        activation  = "sigmoid"     # used to produse probablity
    )
)

# project architecture
# review -> embedding -> LSTM -> Dense -> sigmoid -> either positive or negative

####################################################################################
# step 10  = compile the  model
####################################################################################

model.compile(
    optimizer = " adam",                # algorithm to update weights
    loss = " binary_crossentrophy",     #loss the function
    metrics =["accuracy"]               # measure classificstion accuracy 
)

print("model compile successfully")

####################################################################################
# step 11  = train the  model
####################################################################################

print("model training")
model.fit(
    X_train_padded,           # input training reviews
    Y_train,                  # actual sentiment labels
    epochs = 3,               # complete dataset gets processes 3 times
    batch_size = 64 ,          # process 64 reviews in 1 batch
    validation_split = 0.2    # use 20 percent training for validation
)
print("model training completed")

####################################################################################
# step 12  = Evaluate the model
####################################################################################

accuracy = model.evaluate(
    X_test_padded,      # testing reviews
    Y_test,             # Actual testing labels
    verbose = 0         # Dont display the process bar
)

print("testing accuracy :", accuracy)

####################################################################################
# step 13 = predict the review
####################################################################################

TEST_REVIEW_NUMBER = 0

original_review = X_test[12]
decoded_review = DecodeReview(TEST_REVIEW_NUMBER)

print("Review  given to the model :")
print(decoded_review)

####################################################################################
# step 14 = get the actual sentiment
####################################################################################

actual_value =  Y_test[TEST_REVIEW_NUMBER]

if actual_value == 1:
    actual_sentiment = "positive"
else:
    actual_sentiment = "Negative"
    
print("Actual sentiment :", actual_sentiment)

####################################################################################
# step 15 = predict the sentiment
####################################################################################

review_for_prediction = X_test_padded[TEST_REVIEW_NUMBER: TEST_REVIEW_NUMBER+ 1]

prediction = model.predict(
    review_for_prediction,
    verbose =0
)
probability = prediction[0][0]

if probability >= 0.5:
    predicited_sentiment = "positive"
else:
    predicited_sentiment = " negative"
    
print("final result")

print("prediction probablity:", probablity)
print("Actual Sentiment :", actual_sentiment)
print("predicted sentiment", predicited_sentiment)

print("-"*40)




 


