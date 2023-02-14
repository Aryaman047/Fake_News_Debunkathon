from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

application = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('randomForest.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidf_v.pkl', 'rb'))


# Build functionalities
@application.route('/')
def home():
    return render_template('index.html')


def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction


@application.route('/prediction', methods=['GET', 'POST'])
def webapp():
    if request.method == "POST":
        text = request.form['text']
        prediction = predict(text)
        return render_template('prediction.html', text=text, result=prediction)

    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    application.run(debug=True)
