from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

# Create upload folder
os.makedirs("uploads", exist_ok=True)

# Globals
model = None
tokenizer = None
vgg = None
max_len = 38


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    global model, tokenizer, vgg

    print("Loading caption model...")
    model = load_model("caption_model.h5", compile=False)
    print("✔ caption_model.h5 loaded")

    print("Loading tokenizer...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("✔ tokenizer.pkl loaded")

    print("Loading VGG16 FC2 (4096-dim)...")
    base = VGG16(weights="imagenet", include_top=True)
    vgg = Model(inputs=base.input, outputs=base.get_layer("fc2").output)
    print("✔ VGG16 loaded successfully")


def extract_feature(image_path):
    """Extract 4096-dim features using VGG16 FC2"""
    try:
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = vgg.predict(img, verbose=0)
        return feature
    except Exception as e:
        print("Feature extraction error:", e)
        return None


def idx_to_word(idx):
    for word, word_id in tokenizer.word_index.items():
        if word_id == idx:
            return word
    return None


def generate_caption_greedy(photo):
    """Greedy decoding - picks most probable word at each step"""
    in_text = "startseq"

    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat)
        if word is None or word == "endseq":
            break

        in_text += " " + word

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


def generate_caption_sampling(photo, temperature=0.5):
    """Temperature-based sampling - adds controlled randomness"""
    in_text = "startseq"

    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)

        yhat = model.predict([photo, sequence], verbose=0)
        
        # Apply temperature scaling
        yhat = np.log(yhat + 1e-10) / temperature
        exp_yhat = np.exp(yhat)
        yhat = exp_yhat / np.sum(exp_yhat)
        
        # Sample from probability distribution
        yhat_idx = np.random.choice(len(yhat[0]), p=yhat[0])

        word = idx_to_word(yhat_idx)
        if word is None or word == "endseq":
            break

        in_text += " " + word

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


def generate_caption_no_repeat(photo):
    """Prevents word repetition by masking already used words"""
    in_text = "startseq"
    used_words = set()

    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)

        yhat = model.predict([photo, sequence], verbose=0)[0]
        
        # Mask already used words (except common words)
        common_words = {"a", "an", "the", "in", "on", "at", "of", "to", "and"}
        for word, idx in tokenizer.word_index.items():
            if word in used_words and word not in common_words and idx < len(yhat):
                yhat[idx] = 0
        
        # Re-normalize probabilities
        if yhat.sum() > 0:
            yhat = yhat / yhat.sum()
        
        yhat_idx = np.argmax(yhat)
        word = idx_to_word(yhat_idx)
        
        if word is None or word == "endseq":
            break

        used_words.add(word)
        in_text += " " + word

    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-caption", methods=["POST"])
def generate_caption_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    feature = extract_feature(filepath)
    os.remove(filepath)

    if feature is None:
        return jsonify({"error": "Failed to process image"}), 500

    # Generate all caption types
    try:
        captions = {
            "greedy": generate_caption_greedy(feature),
            "sampling_low": generate_caption_sampling(feature, temperature=0.5),
            "sampling_high": generate_caption_sampling(feature, temperature=0.7),
            "no_repeat": generate_caption_no_repeat(feature)
        }
        
        return jsonify({"success": True, "captions": captions})
    
    except Exception as e:
        print(f"Caption generation error: {e}")
        return jsonify({"error": "Failed to generate captions"}), 500


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)