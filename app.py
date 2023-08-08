from flask import Flask, render_template, request

app = Flask(__name__)

import pickle
import numpy as np

# Load the saved CountVectorizer and model
save_cv = pickle.load(open('count-vectorizer.pkl', 'rb'))
model = pickle.load(open('Movies_review_classification.pkl', 'rb'))

def test_model(sentence):
    sen = save_cv.transform([sentence]).toarray()
    res = model.predict(sen)[0]
    if res == 1:
        return 'positive', '😃'
    else:
        return 'negative', '😔'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    emoji = None
    if request.method == 'POST':
        review = request.form.get('review')
        sentiment, emoji = test_model(review)
        result = f'The review is {sentiment} {emoji}'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
