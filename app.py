from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global objects
model = None
tokenizer = None
vgg = None
max_len = 38


# ------------------------------
# Allowed file
# ------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------
# Load Models ONCE (important)
# ------------------------------
def load_models():
    global model, tokenizer, vgg

    print("Loading models...")

    # 1. Caption model
    model = load_model("caption_model.h5", compile=False)
    print("✔ Caption model loaded")

    # 2. Tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✔ Tokenizer loaded")

    # 3. VGG16 feature extractor (fc2 output = 4096 dims)
    vgg_base = VGG16()
    vgg = Model(inputs=vgg_base.inputs, outputs=vgg_base.layers[-2].output)
    print("✔ VGG16 loaded")

    print("All models are ready!")


# ------------------------------
# Extract VGG16 feature
# ------------------------------
def extract_feature(img_path):
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


# ------------------------------
# Caption Generators
# ------------------------------
def generate_caption_greedy(photo):
    in_text = "startseq"
    for _ in range(max_len):
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
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        preds = model.predict([photo, seq], verbose=0)[0]

        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        yhat_idx = np.random.choice(len(preds), p=preds)

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
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        preds = model.predict([photo, seq], verbose=0)[0]

        # reduce probability of words already used
        for word, idx in tokenizer.word_index.items():
            if word in used and idx < len(preds):
                preds[idx] *= 0.3

        yhat_idx = np.argmax(preds)

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


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    photo = extract_feature(filepath)
    os.remove(filepath)

    if photo is None:
        return jsonify({"error": "Failed to extract features"}), 500

    captions = {
        "greedy": generate_caption_greedy(photo),
        "sampling_low": generate_caption_sampling(photo, 0.5),
        "sampling_high": generate_caption_sampling(photo, 0.8),
        "no_repeat": generate_caption_no_repeat(photo),
    }

    return jsonify({"success": True, "captions": captions})


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
