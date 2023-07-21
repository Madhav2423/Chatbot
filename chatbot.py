
from weather_api import main as get_weather
import io
import random
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 


# nltk.download('punkt') 
# nltk.download('wordnet') 


with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()


sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    global Dragon_response
    Dragon_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        Dragon_response=Dragon_response+"I am sorry! I don't understand you"
        return Dragon_response
    else:
        Dragon_response = Dragon_response+sent_tokens[idx]
        return Dragon_response


flag=True
print("Dragon: My name is Dragon. I will answer your queries. If you want to exit, type Bye!")
while(flag==True):
    user_response = input("User :")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Dragon: You are welcome..")
        elif(user_response == 'weather'):
            user_response = input("Dragon : Please tell the city name :")
            user_response= user_response.lower()
            print(get_weather(user_response))
        else:
            if(greeting(user_response)!=None):
                print("Dragon: "+greeting(user_response))
            else:
                print("Dragon: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    # elif(user_response == 'Weather'):
    #     Dragon_response = "Please tell the city name"
    #     print(get_weather(user_response))
    else:
        flag=False
        print("Dragon: Bye! take care..")    
        
        

