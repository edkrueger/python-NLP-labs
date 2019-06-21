from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

	text = list(set(request.form))[0]
	array = [text]
	predicted_label = clf.predict(array)[0]

	return jsonify(predicted_label)

if __name__ == '__main__':
	
	clf = load("spam_detector.joblib")
	
	app.run()