from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# GLOBALS
model = None
tokenizer = None
vgg = None
max_len = 38

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ---------- HELPERS ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load caption model, tokenizer, and VGG16 locally."""
    global model, tokenizer, vgg

    print("Loading caption model...")
    model = load_model("caption_model.h5", compile=False)
    print("✔ Caption model loaded")

    print("Loading tokenizer...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✔ Tokenizer loaded")

    print("Loading VGG16 local weights...")
    # IMPORTANT: Load VGG16 WITHOUT downloading
    vgg_base = VGG16(weights="vgg16.h5", include_top=False)
    vgg = Model(inputs=vgg_base.input, outputs=vgg_base.layers[-1].output)
    print("✔ VGG16 loaded from local file")
    print("All models ready!")


def extract_feature(img_path):
    """Extract features from image using VGG16"""
    try:
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, 224, 224, 3))
        img = preprocess_input(img)
        feature = vgg.predict(img, verbose=0)
        return feature
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# ---------- CAPTION METHODS ----------
def generate_caption_greedy(photo):
    in_text = "startseq"
    for i in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)
        yhat_idx = np.argmax(yhat)

        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat_idx:
                word = w
                break

        if word is None or word == "endseq":
            break

        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


def generate_caption_sampling(photo, temperature=0.7):
    in_text = "startseq"
    for i in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)[0]

        yhat = np.log(yhat + 1e-10) / temperature
        exp = np.exp(yhat)
        yhat = exp / np.sum(exp)

        yhat_idx = np.random.choice(len(yhat), p=yhat)

        # map index → word
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat_idx:
                word = w
                break

        if word is None or word == "endseq":
            break

        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


def generate_caption_no_repeat(photo):
    in_text = "startseq"
    used = set()

    for i in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)[0]

        # penalize repeated words
        for w, idx in tokenizer.word_index.items():
            if w in used and idx < len(yhat):
                yhat[idx] *= 0.3

        yhat_idx = np.argmax(yhat)

        # map index → word
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat_idx:
                word = w
                break

        if word is None or word == "endseq":
            break

        used.add(word)
        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        photo = extract_feature(path)

        os.remove(path)

        if photo is None:
            return jsonify({"error": "Failed to process image"}), 500

        captions = {
            "greedy": generate_caption_greedy(photo),
            "sampling_low": generate_caption_sampling(photo, 0.5),
            "sampling_high": generate_caption_sampling(photo, 0.7),
            "no_repeat": generate_caption_no_repeat(photo),
        }

        return jsonify({"success": True, "captions": captions})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# ---------- STARTUP ----------
load_models()  # Load all models at startup


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
