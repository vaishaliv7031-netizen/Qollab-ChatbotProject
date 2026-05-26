import os
import re
import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify

# 1. Core NLP & Preprocessing Libraries (NLTK)
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 2. Core Machine Learning & Naive Bayes Libraries (Scikit-Learn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Ensure necessary NLTK components are locally available
for dependency in ['tokenizers/punkt', 'corpora/stopwords', 'corpora/wordnet', 'corpora/punkt_tab']:
    try:
        nltk.data.find(dependency)
    except LookupError:
        # Handle naming variation translation for the download utility
        download_target = dependency.split('/')[-1]
        nltk.download(download_target)

# ------------------ ENVIRONMENT SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT = Flask(__name__, static_folder="static")
CHATBOT.secret_key = "scikit-nb-adaptive-nlp-key"

DATA_FILE_PATH = os.path.join(BASE_DIR, "/home/sonu-nitu/venchat/bin/PROJECT/DATA.json")
CHAT_FILE = os.path.join(BASE_DIR, "/home/sonu-nitu/venchat/bin/PROJECT/memory.json")

# Global data containers for intents and technical categories
data = {}
intents = []
categ = {}

# Initialize NLTK structures
lemmatizer = WordNetLemmatizer()  # Reduces words to their base form
stop_words = set(stopwords.words('english'))
stop_words.update({"want", "learn", "teach", "show", "explain", "about", "tell", "course", "please", "code", "example"})

# Global Machine Learning Pipeline variable
ml_classifier_pipeline = None

# ------------------ NLP PREPROCESSING PIPELINE ------------------

def custom_nltk_tokenizer(text):
    """
    NLP Preprocessing: Tokenizes, cleans punctuation, filters stop-words, 
    and applies lemmatization to extract stable root words.
    """
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    raw_tokens = word_tokenize(text)
    
    cleaned_tokens = []
    for token in raw_tokens:
        if token not in stop_words and len(token) > 1:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return cleaned_tokens

# ------------------ MACHINE LEARNING TRAINING ------------------

def train_naive_bayes_model():
    """
    Model Training Layer: Extracts text patterns, pairs them with intent categories,
    and trains a Scikit-Learn Multinomial Naive Bayes Pipeline on startup.
    """
    global ml_classifier_pipeline, intents
    
    training_sentences = []
    training_labels = []
    
    for intent in intents:
        tag = intent.get("tag")
        patterns = intent.get("patterns", [])
        for pattern in patterns:
            training_sentences.append(pattern)
            training_labels.append(tag)
            
    if training_sentences and training_labels:
        # Build an integrated vectorization and classification pipeline
        ml_classifier_pipeline = make_pipeline(
            TfidfVectorizer(tokenizer=custom_nltk_tokenizer, token_pattern=None, lowercase=False),
            MultinomialNB(alpha=1.0) # Laplace Smoothing applied via alpha=1.0
        )
        # Train the model
        ml_classifier_pipeline.fit(training_sentences, training_labels)
        print("🚀 Naive Bayes Classifier trained successfully using Scikit-Learn!")
    else:
        ml_classifier_pipeline = None

# ------------------ DATA SYSTEM HANDLERS ------------------

def load_data():
    """Loads text profiles and triggers machine learning model training."""
    global data, intents, categ
    try:
        if os.path.exists(DATA_FILE_PATH):
            with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            intents = data.get("intents", [])
            categ = data.get("categories", {})
            
            # Retrain model with any newly configured intent data matrices
            train_naive_bayes_model()
        else:
            data = {"intents": [], "categories": {}}
            intents, categ = [], {}
    except Exception as e:
        print(f"❌ Data load error: {e}")

def load_chat_history():
    if not os.path.exists(CHAT_FILE): 
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except Exception as e:
        print(f"❌ Storage file system access fault: {e}")
        return []

def save_chat_history(chats):
    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=4)
            f.flush()
            os.fsync(f.fileno())  
    except Exception as e: 
        print(f"❌ Storage write failure: {e}")

# Initial system dataset boot load
load_data()

def get_char_ngrams(text, n=3):
    """Breaks text into small letter combinations to tolerate typos."""
    text = f" {text.strip()} "
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def check_semantic_similarity(user_msg, target_strings, threshold=0.2):
    """Checks if the user's input closely overlaps with any targeted strings or patterns."""
    user_grams = get_char_ngrams(user_msg)
    
    for target in target_strings:
        clean_target = target.lower().strip().replace("?", "").replace(",", "")
        target_grams = get_char_ngrams(clean_target)
        
        intersection = user_grams.intersection(target_grams)
        union = user_grams.union(target_grams)
        
        if union:
            similarity = len(intersection) / len(union)
            if similarity >= threshold:
                return True
    return False

def get_bot_response(user_message, chats):
    """Resolves prompts by dynamically digging into sub-categories (like 'Basic') to find topics."""
    load_data()  # Pulls fresh references from DATA.json
    
    # Standardize user input
    clean_msg = user_message.lower().strip().replace("?", "").replace(",", "")
    msg_words = clean_msg.split()
    
    # 🌟 LEVEL 1: CHECK INTENTS
    if intents:
        for intent in intents:
            patterns = intent.get("patterns", [])
            if check_semantic_similarity(clean_msg, patterns, threshold=0.45):
                return random.choice(intent["responses"])

    # 🌟 LEVEL 2: MULTI-LAYERED TOPIC COLLECTOR
    matched_responses = []

    if categ:
        for cat_name, cat_data in categ.items():
            # Create a list of dictionaries to inspect. 
            # We check the main category layer AND any nested sub-layers (like "Basic")
            layers_to_check = [cat_data]
            
            for key, value in cat_data.items():
                if isinstance(value, dict):  # This finds blocks like "Basic", "Advanced", etc.
                    layers_to_check.append(value)
            
            # Now scan all discovered layers for topics
            for layer in layers_to_check:
                topics = layer.get("topics", [])
                
                for topic in topics:
                    title = topic.get("title", "").lower().strip()
                    
                    # 1. Substring containment check
                    # 2. Word-by-word plural protection lookup
                    word_match = any((word in title or title in word) for word in msg_words)
                    # 3. N-gram spelling typo check
                    similarity_match = check_semantic_similarity(clean_msg, [title], threshold=0.40)
                    
                    if title in clean_msg or word_match or similarity_match:
                        # Construct your layout card
                        markdown_response = (
                            f"# 🚀 Topic Found: {title.title()}\n"
                            f"**Curriculum Category:** {cat_name.title()}\n\n"
                            f"**Definition:** *{topic.get('definition', 'No definition provided.')}*\n\n"
                            f"### 📘 Conceptual Breakdown:\n{topic.get('explanation', '')}\n\n"
                        )
                        
                        # Append code windows if they exist
                        examples = topic.get("code_examples", [])
                        if examples:
                            markdown_response += "### 💻 Applied Code Sandbox:\n"
                            for example in examples:
                                markdown_response += f"```\n{example}\n```\n"
                                
                        matched_responses.append(markdown_response)

    # If matches were found across any depth of your JSON, return them combined
    if matched_responses:
        return "\n\n---\n\n".join(matched_responses)

    # ---------------------------------------------------------------------
    # 3. Machine Learning Prediction via Naive Bayes (Fallback security)
    if ml_classifier_pipeline is not None:
        try:
            predicted_tag = ml_classifier_pipeline.predict([user_message])[0]
            probabilities = ml_classifier_pipeline.predict_proba([user_message])[0]
            max_probability = np.max(probabilities)
            
            if max_probability > 0.30 and predicted_tag != "fallback" and predicted_tag != "language_query":
                for intent in intents:
                    if intent.get("tag") == predicted_tag:
                        return random.choice(intent["responses"])
        except Exception as e:
            print(f"⚠️ Classifier inference notice: {e}")
        return " hey👋!.. can you ask something related to my knowledge... i am happy to give you answers 🥰 "
# ------------------ FLASK WEB ROUTING ENDPOINTS ------------------

@CHATBOT.route("/")
def home():
    return render_template("index.html")

@CHATBOT.route("/get_chats", methods=["GET"])
def get_chats(): 
    return jsonify(load_chat_history())

@CHATBOT.route("/save_chat", methods=["POST"])
def save_chat():
    try:
        req_data = request.get_json() or {}
        chat_name = req_data.get("name", "").strip()
        messages = req_data.get("messages", [])
        
        if not chat_name:
            return jsonify({"status": "error", "message": "Missing required text tracking name"}), 400

        chats = load_chat_history()
        for chat in chats:
            if chat.get("name", "").strip() == chat_name:
                chat["messages"] = messages
                save_chat_history(chats)
                return jsonify({"status": "saved", "name": chat_name})
                
        chats.append({"name": chat_name, "messages": messages})
        save_chat_history(chats)
        return jsonify({"status": "saved", "name": chat_name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@CHATBOT.route("/delete-chat", methods=["POST"])
def delete_chat():
    try:
        req_data = request.get_json() or {}
        chat_name = req_data.get("chat_name", "").strip()
        
        if not chat_name:
            return jsonify({"success": False, "message": "Missing routing index data"}), 400
            
        chats = load_chat_history()
        updated_chats = [c for c in chats if c.get("name", "").strip() != chat_name]
        
        save_chat_history(updated_chats)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@CHATBOT.route("/get", methods=["POST"])
def chatbot_api():
    try:
        if not request.is_json:
            return jsonify({"response": "❌ Incompatible data delivery format"}), 400
            
        req_data = request.get_json()
        user_message = req_data.get("message", "").strip()
        chat_name = req_data.get("chat_name", "").strip()

        if not user_message:
            return jsonify({"response": "Please type a message 📝"})

        chats = load_chat_history()
        bot_response = get_bot_response(user_message, chats)
        
        # Dynamic conversational title auto-generation loop
        final_chat_name = chat_name
        if chat_name.startswith("New Chat Thread") or chat_name == "Default Chat":
            words = user_message.split()
            final_chat_name = " ".join(words[:4])
            if len(words) == 1 and words[0].lower() in ["hi", "hello", "hey", "hello!"] or len(words) > 20:
                final_chat_name = "General Discussion"
            else:
                final_chat_name = " ".join(words[:4])
                if len(words) > 20:
                    final_chat_name += "..."
           
            existing_names = [c.get("name", "").strip() for c in chats]
            if final_chat_name in existing_names:
                final_chat_name += f" ({random.randint(10,99)})"

        chat_found = False
        for chat in chats:
            if chat.get("name", "").strip() == chat_name:
                chat["name"] = final_chat_name
                if "messages" not in chat: 
                    chat["messages"] = []
                chat["messages"].append({"user": user_message})
                chat["messages"].append({"bot": bot_response})
                chat_found = True
                break

        if not chat_found:
            chats.append({
                "name": final_chat_name,
                "messages": [{"user": user_message}, {"bot": bot_response}]
            })

        save_chat_history(chats)
        return jsonify({"response": bot_response, "updated_chat_name": final_chat_name})
    except Exception as e:
        return jsonify({"response": f"❌ Flask Pipeline Route Anomaly: {str(e)}"})

if __name__ == "__main__":
    CHATBOT.run(host="0.0.0.0", port=5000, debug=True)
