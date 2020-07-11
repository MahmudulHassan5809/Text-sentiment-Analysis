from flask import Flask, render_template,url_for,request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Variables
app = Flask(__name__)

# Viewss
@app.route('/')
def index():
    return render_template('home.html')


# Views
@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		with open('data/clf.pickle','rb') as f:
			clf = pickle.load(f)
		with open('data/tfidfmodel.pickle','rb') as f:
			tfidf = pickle.load(f)
		text = request.form['text']
		data = [text]
		vect = tfidf.transform(data).toarray()
		my_prediction = clf.predict(vect)
		return render_template('home.html',predction = my_prediction)
	return render_template('home.html')

# Run
if __name__ == '__main__':
    app.run(debug=True)
