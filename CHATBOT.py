import os
import re
import json
import random
import numpy as np
from flask import Flask, render_template, request, jsonify, session 

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
        download_target = dependency.split('/')[-1]
        nltk.download(download_target)

# ------------------ ENVIRONMENT SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT = Flask(__name__, static_folder="static")
CHATBOT.secret_key = "scikit-nb-adaptive-nlp-key"

# Fixed Absolute File Paths
DATA_FILE_PATH = os.path.join(BASE_DIR, "DATA.json")
CHAT_FILE = os.path.join(BASE_DIR, "memory.json")

# Global data containers for intents and technical categories
data = {}
intents = []
categ = {}

# Initialize NLTK structures
lemmatizer = WordNetLemmatizer()  
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
        ml_classifier_pipeline = make_pipeline(
            TfidfVectorizer(tokenizer=custom_nltk_tokenizer, token_pattern=None, lowercase=False),
            MultinomialNB(alpha=1.0)
        )
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
            
            # Safe checking if root data is a list wrapper or raw dictionary object
            if isinstance(data, list) and len(data) > 0:
                root_item = data[0]
            else:
                root_item = data

            intents = root_item.get("intents", [])
            categ = root_item.get("categories", {})
            
            train_naive_bayes_model()
        else:
            data = {"intents": [], "categories": {}}
            intents, categ = [], {}
    except Exception as e:
        print(f"❌ Data load error: {e}")

# Private browser session history control arrays
def load_chat_history():
    """Loads a private chat history array unique to the current user's browser session."""
    if 'history' not in session:
        session['history'] = []
    return session['history']

def save_chat_history(chats):
    """Saves the chat array into the user's private session container."""
    session['history'] = chats
    session.modified = True  

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

# ------------------ DEEP NESTED OBJECT TRAVERSAL ENGINE ------------------
def get_bot_response(user_message, database_json):
    """
    Precision-Ordered Retrieval Engine:
    1. Intercepts short keyword topics directly to prevent ML confusion.
    2. Runs conversational ML/Pattern checks ONLY for greetings, help, and exit intents.
    3. Searches multi-category dictionary fields for technical topics and code sandboxes.
    """
    global ml_classifier_pipeline, intents
    user_query = user_message.lower().strip()
    fallback_response = "hey👋!.. can you ask something related to my knowledge... i am happy to give you answers 🥰"
    
    # Secure database block from list wrappers safely
    if isinstance(database_json, list) and len(database_json) > 0:
        root_data = database_json[0]
    else:
        root_data = database_json

    # ⚡ STEP 1: KEYWORD INTERCEPT (Fixes the "python" / "java" generic mismatch bug)
    if user_query in ["python", "learn python", "teach me python"]:
        return "### 🐍 Python Track Active\nPython is a powerful, high-level language focused on code readability. Try asking me specific concepts like:\n* 👉 *'python syntax and indentation'*\n* 👉 *'list and sequence mastery'*\n* 👉 *'dictionaries and mapping'*"
    
    if user_query in ["java", "learn java", "i want to learn java"]:
        return "### ☕ Java Track Active\nJava is a strongly-typed, object-oriented language used worldwide. Try asking me specific concepts like:\n* 👉 *'objects and classes'*\n* 👉 *'conditional logic'*\n* 👉 *'loops and iteration'*"

    if user_query in ["c", "c programming", "what about c programming"]:
        return "### 💻 C Track Active\nC is a foundational system-level language that gives you complete power over memory allocation. Try asking me about:\n* 👉 *'memory management'*\n* 👉 *'arrays and collections'*"

    # 🤖 STEP 2: CONVERSATIONAL INTENTS ONLY (Greetings, Help, Goodbye, Thanks)
    # This prevents your ML model from overriding technical coding keywords!
    if ml_classifier_pipeline is not None:
        try:
            predicted_tag = ml_classifier_pipeline.predict([user_message])[0]
            probabilities = ml_classifier_pipeline.predict_proba([user_message])[0]
            
            # Only trigger ML if it's a known conversational tag and confidence is high
            if predicted_tag in ["greeting", "help", "goodbye", "thanks", "response"] and np.max(probabilities) > 0.50:
                for intent in intents:
                    if intent.get("tag") == predicted_tag:
                        return random.choice(intent.get("responses"))
        except Exception as e:
            print(f"⚠️ ML optimization bypass: {e}")

    # 📁 STEP 3: BACKUP FLAT PATTERN MATCHING (For greetings/help without high ML confidence)
    intents_list = root_data.get("intents", [])
    for intent in intents_list:
        if intent.get("tag") in ["greeting", "help", "goodbye", "thanks", "response"]:
            patterns = intent.get("patterns", [])
            if check_semantic_similarity(user_query, patterns, threshold=0.30):
                return random.choice(intent.get("responses", [fallback_response]))

    # 🗂️ STEP 4: DYNAMIC MULTI-CATEGORY TOPIC SEARCH
    all_categories = root_data.get("categories", {})
    for category_track_name, category_content in all_categories.items():
        if not isinstance(category_content, dict):
            continue
            
        # Check tracking names or history blocks
        if f"history of {category_track_name.lower()}" in user_query or (category_track_name.lower() in user_query and "history" in user_query):
            return f"# 📜 History of {category_track_name}\n{category_content.get('history')}"

        # Loop through dynamic structural levels (Basic, Intermediate, Advanced, python Basic, etc.)
        for level_key, level_value in category_content.items():
            if isinstance(level_value, dict) and "topics" in level_value:
                topics_list = level_value.get("topics", [])
                
                for topic in topics_list:
                    title = topic.get("title", "").lower()
                    intro = topic.get("intro", "").lower()
                    
                    # Search list metrics to clear out spelling typos (like "memmory")
                    search_matrix = [title, intro]
                    
                    if (check_semantic_similarity(user_query, search_matrix, threshold=0.22) or 
                        title in user_query or user_query in title):
                        
                        raw_explanation = topic.get("explanation", "")
                        
                        # Gather code sandbox elements
                        code_blocks = ""
                        examples = topic.get("code_examples", [])
                        for example in examples:
                            lang_tag = "python" if "python" in category_track_name.lower() else "c"
                            code_blocks += f"\n\`\`\`{lang_tag}\n{example}\n\`\`\`\n"
                        
                        formatted_response = (
                            f"# 🚀 Topic Found: {topic.get('title').title()}\n"
                            f"**Track:** {category_track_name.title()} — [{level_key}]\n\n"
                            f"**Introduction:** *{topic.get('intro')}*\n\n"
                            f"### 📘 Conceptual Breakdown:\n{raw_explanation}\n"
                        )
                        if code_blocks:
                            formatted_response += f"\n### 💻 Applied Code Sandbox:{code_blocks}"
                            
                        return formatted_response

    # 🛑 STEP 5: SAFE ACCESSIBLE FALLBACK
    return fallback_response
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
        
        # 🌟 FIXED: Passing 'data' global textbook variable instead of user chat log list
        bot_response = get_bot_response(user_message, data)
        
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
