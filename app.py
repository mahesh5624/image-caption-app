from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}

# GLOBALS
model = None
tokenizer = None
vgg = None
max_len = 38


# ---------- ALLOWED FILE ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------- LOAD MODELS AT STARTUP ----------
def load_all_models():
    global model, tokenizer, vgg

    print("Loading Caption Model...")
    model = load_model("caption_model.h5", compile=False)

    print("Loading Tokenizer...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    print("Loading VGG16...")
    base = VGG16(weights="imagenet")
    vgg = Model(inputs=base.input, outputs=base.layers[-2].output)

    print("✔ All models loaded successfully!")


load_all_models()


# ---------- FEATURE EXTRACTION ----------
def extract_features(img_path):
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


# ---------- CAPTION GENERATION ----------
def greedy_caption(photo):
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

    return in_text.replace("startseq", "").strip()


# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    photo = extract_features(filepath)
    os.remove(filepath)

    if photo is None:
        return jsonify({"error": "Failed to process image"}), 500

    caption = greedy_caption(photo)

    return jsonify({"success": True, "caption": caption})


# ---------- RENDER REQUIRED ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
