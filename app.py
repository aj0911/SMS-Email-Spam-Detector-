import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle
import streamlit as st
tfid = pickle.load(open('vectorizer.pkl','rb'))
mnb = pickle.load(open('model.pkl','rb'))

def text_transformation(text):
    #lower case
    text = text.lower();

    # tokenization
    text = nltk.word_tokenize(text);
    
    #Removing Special Characters
    y = [];
    for i in text:
        if i.isalnum():
            y.append(i);
    text  = y[:];
    y.clear();

    #Removing Punctuation and stop words
    for i in text:
        if i  not in stopwords.words('english') and i not in string.punctuation:
            y.append(i);
    text = y[:];
    y.clear();

    #Stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i));
    text = ' '.join(y);
    y.clear();

    return text;

def predict_spam(input_sms):
    # 1. preprocess
    transformed_sms = text_transformation(input_sms)
    # 2. vectorize
    vector_input = tfid.transform([transformed_sms])
    # 3. predict
    result = mnb.predict(vector_input)[0]
    return result;

st.title('Email/ SMS Spam Classifier')

inputSms = st.text_area(label='Message Field',placeholder='Enter the message',height=150);

if st.button('Predict'):
    if inputSms=='':
        st.warning('SMS is mandatory');
    else:
        res = predict_spam(inputSms);
        if(res==0):
            st.success('Not Spam Message')
        else:
            st.error('Spam Message');