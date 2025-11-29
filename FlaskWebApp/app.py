from flask import Flask, render_template, request
import pickle

tokenizer = pickle.load(open(r'D:\\python projects\\MachineLearning\\FlaskWebApp\\models\\cv.pkl', "rb"))
model = pickle.load(open(r'D:\\python projects\\MachineLearning\\FlaskWebApp\\models\\clf.pkl', "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    email_text = request.form.get('email_content')

    if not email_text or email_text.strip() == "":
        return render_template("index.html", error="Please enter email text.")

    tokenized_email = tokenizer.transform([email_text])

    prediction = model.predict(tokenized_email)[0]

    # Spam probability
    try:
        prob = model.predict_proba(tokenized_email)[0][1]
        spam_prob = round(prob * 100, 2)
    except:
        spam_prob = None

    predictions = 1 if prediction == 1 else -1

    return render_template(
        "index.html",
        predictions=predictions,
        spam_prob=spam_prob,
        email_text=email_text,
        is_spam=True if predictions == 1 else False
    )


if __name__== '__main__':
    app.run(debug=True)
