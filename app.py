from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import logging
import traceback

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from werkzeug.utils import secure_filename

# --- Set up logging ---
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

# Create upload folder in /tmp for serverless environments
UPLOAD_DIR = os.path.join("/tmp", "uploads") if os.path.exists("/tmp") else "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)
logging.info(f"Using upload directory: {UPLOAD_DIR}")

# Globals
model = None
tokenizer = None
vgg = None
max_len = 38


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Updated load_models with error logging ---
def load_models():
    global model, tokenizer, vgg

    try:
        logging.info("Loading caption model (caption_model.h5)...")
        # This will load your RENAMED epoch_20 model
        model = load_model("caption_model.h5", compile=False)
        logging.info("✔ caption_model.h5 loaded")
    except FileNotFoundError:
        logging.error("❌ ERROR: caption_model.h5 not found.")
    except MemoryError:
        logging.error("❌ MEMORY ERROR: Ran out of RAM loading caption_model.h5.")
    except Exception as e:
        logging.error(f"❌ Unknown error loading caption_model.h5: {e}")
        traceback.print_exc()

    try:
        logging.info("Loading tokenizer (tokenizer.pkl)...")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        logging.info("✔ tokenizer.pkl loaded")
    except FileNotFoundError:
        logging.error("❌ ERROR: tokenizer.pkl not found.")
    except Exception as e:
        logging.error(f"❌ Unknown error loading tokenizer.pkl: {e}")
        traceback.print_exc()

    try:
        logging.info("Loading VGG16 FC2 (4096-dim)...")
        base = VGG16(weights="imagenet", include_top=True)
        vgg = Model(inputs=base.input, outputs=base.get_layer("fc2").output)
        logging.info("✔ VGG16 loaded successfully")
    except MemoryError:
        logging.error("❌ MEMORY ERROR: Ran out of RAM loading VGG16.")
    except Exception as e:
        logging.error(f"❌ Unknown error loading VGG16: {e}")
        traceback.print_exc()


def extract_feature(image_path):
    """Extract 4096-dim features using VGG16 FC2"""
    if vgg is None:
        logging.error("VGG model is None. Cannot extract features.")
        return None
        
    try:
        logging.info(f"Processing image: {image_path}")
        if not os.path.exists(image_path):
            logging.error(f"Error: Image file not found at {image_path}")
            return None
            
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        logging.info("Extracting features with VGG16...")
        feature = vgg.predict(img, verbose=0)
        logging.info(f"Feature shape: {feature.shape}")
        return feature
    except Exception as e:
        logging.error(f"Feature extraction error: {e}")
        traceback.print_exc()
        return None


def idx_to_word(idx):
    if tokenizer is None:
        return None
    for word, word_id in tokenizer.word_index.items():
        if word_id == idx:
            return word
    return None


def generate_caption_greedy(photo):
    """Greedy decoding - picks most probable word at each step"""
    if model is None or tokenizer is None:
        return "Error: Model not loaded."
        
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
    return in_text.replace("startseq", "").replace("endseq", "").strip()


# --- DELETED THE OTHER 2 GENERATE FUNCTIONS ---


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate-caption", methods=["POST"])
def generate_caption_route():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        import time
        timestamp = str(int(time.time() * 1000))
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        logging.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        if not os.path.exists(filepath):
            logging.error("File save failed, os.path.exists is false.")
            return jsonify({"error": "Failed to save file"}), 500

        logging.info(f"File saved successfully, size: {os.path.getsize(filepath)} bytes")
        
        feature = extract_feature(filepath)
        
        try:
            os.remove(filepath)
            logging.info(f"Temporary file removed: {filepath}")
        except Exception as e:
            logging.warning(f"Could not remove temp file: {e}")

        if feature is None:
            logging.error("Feature extraction returned None. Check model loading logs.")
            return jsonify({"error": "Failed to process image. Model may not be loaded."}), 500

        logging.info("Generating caption...")
        
        # --- CHANGED: Only generate the one best caption ---
        caption = generate_caption_greedy(feature)
        captions = {
            "greedy": caption
        }
        # --- END CHANGE ---
        
        logging.info(f"Captions generated: {captions}")
        return jsonify({"success": True, "captions": captions})
    
    except Exception as e:
        logging.error(f"Unknown error in /generate-caption route: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# --- FIX: Call load_models() at global scope ---
# This ensures it runs when Gunicorn (Render) starts the app.
load_models()


# This block is for local testing (running `python app.py`)
if __name__ == "__main__":
    logging.info("Starting Flask app in debug mode for local testing...")
    app.run(host="0.0.0.0", port=5000, debug=True)
