
import os

import re

import json

import random

import uuid

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



def load_chat_history():

    if 'history' not in session:

        session['history'] = []

    return session['history']



def save_chat_history(chats):

    session['history'] = chats

    session.modified = True  



# Initial system dataset boot load

load_data()



def get_char_ngrams(text, n=3):

    """Breaks text into small letter combinations to tolerate typos."""

    text = f" {text.strip()} "

    return set(text[i:i+n] for i in range(len(text) - n + 1))



def check_semantic_similarity(user_msg, target_strings, threshold=0.22):

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



# ------------------ UNTOUCHED ORIGINAL METHOD ------------------

def get_bot_response(user_message, database_json):

    """

    Precision-Ordered Retrieval Engine: Preserves your original lookups perfectly.

    """

    global ml_classifier_pipeline, intents

    user_query = user_message.lower().strip()

    fallback_response = "hey👋!.. can you ask something related to my knowledge... i am happy to give you answers 🥰"

    

    if isinstance(database_json, list) and len(database_json) > 4:

        root_data = database_json[0]

    else:

        root_data = database_json



    if user_query in ["python", "learn python", "teach me python"]:

        return "### 🐍 Python Track Active\nPython is a powerful, high-level language focused on code readability. Try asking me specific concepts like:\n* 👉 *'python syntax and indentation'*\n* 👉 *'list and sequence mastery'*\n* 👉 *'dictionaries and mapping'*"

    

    if user_query in ["java", "learn java", "i want to learn java"]:

        return "### ☕ Java Track Active\nJava is a strongly-typed, object-oriented language used worldwide. Try asking me specific concepts like:\n* 👉 *'objects and classes'*\n* 👉 *'conditional logic'*\n* 👉 *'loops and iteration'*"



    if user_query in ["c", "c programming", "what about c programming"]:

        return "### 💻 C Track Active\nC is a foundational system-level language that gives you complete power over memory allocation. Try asking me about:\n* 👉 *'memory management'*\n* 👉 *'arrays and collections'*"



    if ml_classifier_pipeline is not None:

        try:

            predicted_tag = ml_classifier_pipeline.predict([user_message])[0]

            probabilities = ml_classifier_pipeline.predict_proba([user_message])[0]

            

            if predicted_tag in ["greeting", "help", "goodbye", "thanks", "response"] and np.max(probabilities) > 0.50:

                for intent in intents:

                    if intent.get("tag") == predicted_tag:

                        return random.choice(intent.get("responses"))

        except Exception as e:

            print(f"⚠️ ML optimization bypass: {e}")



    intents_list = root_data.get("intents", [])

    for intent in intents_list:

        if intent.get("tag") in ["greeting", "help", "goodbye", "thanks", "response"]:

            patterns = intent.get("patterns", [])

            if check_semantic_similarity(user_query, patterns, threshold=0.30):

                return random.choice(intent.get("responses", [fallback_response]))



    all_categories = root_data.get("categories", {})

    for category_track_name, category_content in all_categories.items():

        if not isinstance(category_content, dict):

            continue

            

        if f"history of {category_track_name.lower()}" in user_query or (category_track_name.lower() in user_query and "history" in user_query):

            return f"# 📜 History of {category_track_name}\n{category_content.get('history')}"



        for level_key, level_value in category_content.items():

            if isinstance(level_value, dict) and "topics" in level_value:

                topics_list = level_value.get("topics", [])

                

                for topic in topics_list:

                    title = topic.get("title", "").lower()

                    intro = topic.get("intro", "").lower()

                    

                    search_matrix = [title, intro]

                    

                    if (check_semantic_similarity(user_query, search_matrix, threshold=0.22) or 

                        title in user_query or user_query in title):

                        

                        raw_explanation = topic.get("explanation", "")

                        

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



    return fallback_response



# ------------------ ENHANCED COMBINATORIAL SUPERVISOR LAYER ------------------
def process_combined_multi_queries(user_message, database_json):

    clean_msg = user_message.lower().strip()

    

    # --- ADD THIS FAST-TRACK BLOCK ---

    # Check for simple greetings or common non-technical interactions first

    if clean_msg in ["hi", "hello", "hey", "hello!", "hi!"]:

        return get_bot_response(user_message, database_json)

    # ---------------------------------



    root_data = database_json[0] if isinstance(database_json, list) else database_json

    all_categories = root_data.get("categories", {})

    # ... rest of your code stays the same

    found_sections = []

    

    # 1. Scanning Phase: Loop through everything to find matches

    for cat_name, cat_content in all_categories.items():

        if not isinstance(cat_content, dict): continue

        for level_key, level_val in cat_content.items():

            if isinstance(level_val, dict) and "topics" in level_val:

                for topic in level_val.get("topics", []):

                    title = topic.get("title", "").lower()

                    if clean_msg == title or title in clean_msg:

                        found_sections.append(topic)

                    elif len(clean_msg) > 3 and any(word == clean_msg for word in title.split()):

                        found_sections.append(topic)



    # 2. Response Phase: Build the response ONLY after the scan is complete

    if found_sections:

        # Cap the results to prevent spamming the user

        if len(found_sections) > 3:

            found_sections = found_sections[:3]

            

        response_parts = [f"# 🚀 Combined Learning Results"]

        for topic in found_sections:

            title = topic.get('title', 'Topic').title()

            intro = topic.get('intro', '')

            explanation = topic.get('explanation', '')

            part = f"\n### 📘 {title}\n*{intro}*\n{explanation}\n"

            examples = topic.get("code_examples", [])

            if examples:

                part += "\n**Code Example:**\n```python\n" + "\n".join(examples) + "\n```"

            response_parts.append(part)

        

        return "\n---\n".join(response_parts)



    # 3. Fallback: Only hit the original engine if no topics were found

    return get_bot_response(user_message, database_json)
# ------------------ FLASK WEB ROUTING ENDPOINTS ------------------



@CHATBOT.route("/")

def home():

    if "user_id" not in session:

        session["user_id"] = str(uuid.uuid4())

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

        

        bot_response = process_combined_multi_queries(user_message, data)

        

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

    CHATBOT.run(host="0.0.0.0", port=5000)
