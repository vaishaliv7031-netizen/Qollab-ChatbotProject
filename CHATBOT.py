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

# Ensure necessary NLTK components are locally available
# Added 'corpora/punkt_tab' to fix the Render tokenization crash
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

# ------------------ ADAPTIVE & SEMANTIC LEARNING ------------------

def calculate_adaptive_weight(topic_title, chats):
    """
    Adaptive Learning Module: Evaluates conversation history patterns 
    to dynamically shift topic selection biases based on past usage.
    """
    interaction_weight = 1.0
    for chat in chats:
        for msg in chat.get("messages", []):
            bot_text = msg.get("bot", "")
            if f"Topic Found: {topic_title.title()}" in bot_text:
                interaction_weight += 0.20 # Step up selection probability
    return min(interaction_weight, 3.0)

def semantic_curriculum_search(user_query, chats):
    """
    Semantic Learning Vector Map: Evaluates keyword groups using 
    synonym expansions, language limits, and adaptive weights.
    """
    query_tokens = set(custom_nltk_tokenizer(user_query))
    if not query_tokens:
        return None

    target_lang = None
    if "python" in query_tokens: target_lang = "python"
    elif "java" in query_tokens: target_lang = "java"
    elif "c" in query_tokens: target_lang = "c"

    concept_synonyms = {
        "loop": {"loop", "iteration", "for", "while", "repeat", "condition"},
        "variable": {"variable", "scope", "memory", "type", "storage", "allocate"},
        "function": {"function", "method", "modular", "logic", "def", "return"},
        "tuple": {"tuple", "sequence", "immutable", "structure"},
        "list": {"list", "sequence", "collection", "append", "array"},
        "operator": {"operator", "arithmetic", "math", "logic", "calculate"}
    }

    expanded_tokens = set(query_tokens)
    for core_concept, structural_words in concept_synonyms.items():
        if core_concept in query_tokens:
            expanded_tokens.update(structural_words)

    best_topic = None
    highest_score = 0.0
    matched_cat = ""
    matched_lvl = ""

    for cat_name, cat_data in categ.items():
        if not isinstance(cat_data, dict):
            continue
        
        lang_prefix = cat_name.lower().split()[0]
        lang_multiplier = 1.0
        
        if target_lang:
            if target_lang == lang_prefix:
                lang_multiplier = 3.5  
            else:
                lang_multiplier = 0.05 

        for level_name, level_data in cat_data.items():
            if not isinstance(level_data, dict) or "topics" not in level_data:
                continue
                
            for topic in level_data["topics"]:
                title_tokens = set(custom_nltk_tokenizer(topic.get("title", "")))
                definition_tokens = set(custom_nltk_tokenizer(topic.get("definition", "")))
                
                title_hits = len(expanded_tokens.intersection(title_tokens))
                def_hits = len(expanded_tokens.intersection(definition_tokens))
                
                base_score = (title_hits * 35) + (def_hits * 12)
                
                if topic.get("title", "").lower().strip() in user_query.lower():
                    base_score += 90
                
                final_score = base_score * lang_multiplier
                adaptive_multiplier = calculate_adaptive_weight(topic.get("title", ""), chats)
                final_score *= adaptive_multiplier

                if final_score > highest_score and base_score > 0:
                    highest_score = final_score
                    best_topic = topic
                    matched_cat = cat_name
                    matched_lvl = level_name

    # Handle Category summaries if specific search token weight is minimal
    if highest_score < 20:
        for cat_name, cat_data in categ.items():
            lang_prefix = cat_name.lower().split()[0]
            if lang_prefix in query_tokens and len(query_tokens) <= 2:
                response = f"# 🚀 Category Profile: {cat_name.title()}\n\n"
                if "description" in cat_data:
                    response += f"**Core Concept:** {cat_data['description']}\n\n"
                if "history" in cat_data:
                    response += f"**Historical Context:** {cat_data['history']}\n\n"
                return response.strip()

    # Build topic content output structure
    if best_topic and highest_score >= 15:
        title = best_topic.get("title", "").title()
        definition = best_topic.get("definition", "")
        explanation = best_topic.get("explanation", "")
        
        response = f"# 🚀 Topic Found: {title}\n"
        response += f"**Curriculum Area:** {matched_cat.title()} — [{matched_lvl.title()} Track]\n\n"
        response += f"**Definition:** *{definition}*\n\n"
        if explanation:
            response += f"### 📘 Conceptual Breakdown:\n{explanation}\n\n"
            
        examples = best_topic.get("code_examples", [])
        if examples:
            response += "### 💻 Applied Code Sandbox:\n"
            for idx, code_snippet in enumerate(examples, 1):
                response += f"```javascript\n// Functional Sample #{idx}\n{code_snippet}\n```\n\n"
        return response.strip()

    return None

def get_bot_response(user_message, chats):
    
    """Resolves prompts through predictive models and curriculum match rules."""
    load_data()
    
    # Clean the input message
    clean_msg = user_message.lower().strip()
    
    # 🌟 FIX: Catch short filler/affirmation words immediately 
    if clean_msg in ["yes", "sure", "ok", "yep", "yeah", "okay"]:
        return "Awesome! Let me know what programming topic or language concept you want to explore next, or type something like 'explain functions in python'!"

    # Direct exact-match fallback for simple greetings
    for intent in intents:
        if clean_msg in [pattern.lower().strip() for pattern in intent.get("patterns", [])]:
            return random.choice(intent["responses"])
            
    # 1. Machine Learning Prediction via Naive Bayes (Scikit-Learn Pipeline)
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

    # 2. Adaptive Semantic Data Mapping Fallback
    curriculum_match = semantic_curriculum_search(user_message, chats)
    if curriculum_match:
        return curriculum_match

    return "🤖 I couldn't reliably map your query with my machine learning classifier. Try framing it explicitly like: 'explain functions in python'."
# ------------------ SERVER ENDPOINTS & CHANNELS ------------------

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
                if len(words) > 4:
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
