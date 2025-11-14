from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model vars
model = None
tokenizer = None
vgg = None
max_len = 38


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load caption model, tokenizer, and VGG16 at server start."""
    global model, tokenizer, vgg

    print("\n======== LOADING MODELS ========\n")

    # Load trained caption model
    model = load_model("caption_model.h5", compile=False)
    print("✔ Loaded caption model")

    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✔ Loaded tokenizer")

    # Load VGG16 feature extractor
    vgg_model = VGG16()
    vgg = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    print("✔ Loaded VGG16")

    print("\n======== MODELS READY ========\n")


# 🚀 IMPORTANT: Load models ONCE at startup
load_models()


def extract_feature(img_path):
    """Extract features using VGG16."""
    try:
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, 224, 224, 3))
        img = preprocess_input(img)
        feature = vgg.predict(img, verbose=0)
        return feature
    except Exception as e:
        print("Feature extraction error:", e)
        return None


def generate_caption_greedy(photo):
    """Greedy decoding."""
    in_text = "startseq"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)

        yhat = model.predict([photo, seq], verbose=0)
        yhat_idx = np.argmax(yhat)

        word = next((w for w, i in tokenizer.word_index.items() if i == yhat_idx), None)

        if word is None or word == "endseq":
            break

        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


def generate_caption_sampling(photo, temperature=0.7):
    """Temperature sampling."""
    in_text = "startseq"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)

        yhat = model.predict([photo, seq], verbose=0)[0]

        yhat = np.log(yhat + 1e-10) / temperature
        yhat = np.exp(yhat) / np.sum(np.exp(yhat))

        yhat_idx = np.random.choice(len(yhat), p=yhat)

        word = next((w for w, i in tokenizer.word_index.items() if i == yhat_idx), None)

        if word is None or word == "endseq":
            break

        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


def generate_caption_no_repeat(photo):
    """Block repeated words."""
    in_text = "startseq"
    used = set()

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)

        yhat = model.predict([photo, seq], verbose=0)[0]

        for w, i in tokenizer.word_index.items():
            if w in used and i < len(yhat):
                yhat[i] *= 0.3

        yhat_idx = np.argmax(yhat)

        word = next((w for w, i in tokenizer.word_index.items() if i == yhat_idx), None)

        if word is None or word == "endseq":
            break

        used.add(word)
        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        photo = extract_feature(filepath)

        if photo is None:
            return jsonify({'error': 'Failed to process image'}), 500

        captions = {
            'greedy': generate_caption_greedy(photo),
            'sampling_low': generate_caption_sampling(photo, 0.5),
            'sampling_high': generate_caption_sampling(photo, 0.7),
            'no_repeat': generate_caption_no_repeat(photo)
        }

        os.remove(filepath)

        return jsonify({'success': True, 'captions': captions})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
