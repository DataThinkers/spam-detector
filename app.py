import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    #Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier


    clf = MultinomialNB()
    clf.fit(X, y)
    pickle.dump(clf,open('model_spam.pkl','wb'))
    model = pickle.load(open('model_spam.pkl', 'rb'))
    '''
    For rendering results on HTML GUI
    '''
    message = request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    prediction = model.predict(vect)
    # int_features = [str(x) for x in request.form.values()]
    # data=[int_features]
    # final_features = cv.transform(data).toarray()
    # prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    if output == 1:
        res_val = "** SPAM **"
    else:
        res_val = "** Not a Spam **"
    

    return render_template('index.html', prediction_text='Message is :: {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=False)
