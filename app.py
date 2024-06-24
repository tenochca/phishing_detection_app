from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from keras.preprocessing.sequence import pad_sequences
import pickle

#loading in the model
model = load_model('model.keras')

#loading in the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def prep_data(subject, body):
    'prep subject and body txt data'
    data = [subject + ' ' + body]
    seq = tokenizer.texts_to_sequences(data)
    X = pad_sequences(seq, maxlen=100)
    return X

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        subject = request.form.get('sline')
        body = request.form.get('body')

        data = prep_data(subject, body)
        pred = model.predict(data)
        label = [1 if prob > 0.5 else 0 for prob in pred]

        if label[0] == 1:
            return "SPAM"
        elif label[0] == 0:
            return "OK"
        else:
            print(pred)
            print(label)
            return "FAIL"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
