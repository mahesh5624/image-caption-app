from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array   # ⭐ FIXED IMPORT
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
model = None
tokenizer = None
vgg = None
max_len = 38


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load captioning model, tokenizer and VGG16"""
    global model, tokenizer, vgg

    print("Loading caption model...")
    model = load_model("caption_model.h5", compile=False)

    print("Loading tokenizer...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading VGG16 feature extractor...")
    vgg16 = VGG16()
    vgg = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)

    print("All models loaded successfully!")


def extract_feature(img_path):
    """Extract image features using VGG16"""
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
    """Greedy decoding caption generation"""
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
    """Temperature sampling caption"""
    in_text = "startseq"

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)[0]

        yhat = np.log(yhat + 1e-10) / temperature
        yhat = np.exp(yhat) / np.sum(np.exp(yhat))

        yhat_idx = np.random.choice(len(yhat), p=yhat)

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
    """Caption with repetition penalty"""
    in_text = "startseq"
    used_words = set()

    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)[0]

        # penalty
        for w, idx in tokenizer.word_index.items():
            if w in used_words and idx < len(yhat):
                yhat[idx] *= 0.3

        yhat_idx = np.argmax(yhat)

        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat_idx:
                word = w
                break

        if word is None or word == "endseq":
            break

        used_words.add(word)
        in_text += " " + word

    return in_text.replace("startseq", "").replace("endseq", "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        photo = extract_feature(filepath)

        if photo is None:
            return jsonify({"error": "Image processing failed"}), 500

        captions = {
            "greedy": generate_caption_greedy(photo),
            "sampling_low": generate_caption_sampling(photo, 0.5),
            "sampling_high": generate_caption_sampling(photo, 0.7),
            "no_repeat": generate_caption_no_repeat(photo)
        }

        os.remove(filepath)

        return jsonify({"success": True, "captions": captions})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_models()  # ⭐ Load everything BEFORE starting server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
