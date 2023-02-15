from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
import re
import nltk
import pickle
import sklearn
import numpy
nltk.download('wordnet')

application = Flask(__name__)
lemmatizer = nltk.stem.WordNetLemmatizer()
# Load model and vectorizer

try:

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = pickle.load(open('LSTM.pkl', 'rb'))
    w2v = pickle.load(open('w2v_tokenizer.pkl', 'rb'))

    def predict(text):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        review_vect = w2v.texts_to_sequences([review])
        review_arr = numpy.asarray(review_vect)
        print(review_arr)
        review_pad = pad_sequences(review_arr, maxlen=1000)
        prediction = 'FAKE' if model.predict(review_pad) <= 0.5 else 'REAL'
        return prediction

except:
    # Load model and vectorizer
    model = pickle.load(open('randomForest.pkl', 'rb'))
    tfidfvect = pickle.load(open('tfidf_v.pkl', 'rb'))


    def predict(text):
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        review_vect = tfidfvect.transform([review]).toarray()
        prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
        return prediction

# Build functionalities
@application.route('/')
def home():
    return render_template('index.html')


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