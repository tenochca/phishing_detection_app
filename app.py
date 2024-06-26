from flask import Flask, render_template, request, flash, redirect, url_for
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from keras.preprocessing.sequence import pad_sequences
import pickle

#loading in the model
model = load_model('model.keras')

#loading in the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def prep_data(sender, subject, body):
    'prep subject and body txt data'
    data = [sender + ' ' + subject + ' ' + body]
    seq = tokenizer.texts_to_sequences(data)
    X = pad_sequences(seq, maxlen=100)
    return X

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sender = request.form.get('sender')
        subject = request.form.get('sline')
        body = request.form.get('body')
        if body == '' or sender == '' or subject == '':
            return "Missing 1 (or more) required fields\n Please use the back arrow to return to classifier"
        data = prep_data(sender, subject, body)
        pred = model.predict(data)
        label = [1 if prob > 0.5 else 0 for prob in pred]

        if label[0] == 1:
            return render_template('scam.html')
        elif label[0] == 0:
            return render_template('okay.html')
        else:
            print(pred)
            print(label)
            return "FAIL"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
