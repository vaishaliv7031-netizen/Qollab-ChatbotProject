import os
import re
import json
import random
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify

# ------------------ ENVIRONMENT SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=dotenv_path)

CHATBOT = Flask(__name__, static_folder="static")
CHATBOT.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-fallback-key")

DATA_FILE_PATH = os.path.join(BASE_DIR, "DATA.json")
CHAT_FILE = os.path.join(BASE_DIR, "memory.json")

data = {}
intents = []
categ = {}

def load_data():
    global data, intents, categ
    try:
        if os.path.exists(DATA_FILE_PATH):
            with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            intents = data.get("intents", [])
            categ = data.get("categories", {})
        else:
            data = {"intents": [], "categories": {}}
            intents, categ = [], {}
    except Exception as e:
        print(f"❌ Error parsing DATA.json: {e}")
        data = {"intents": [], "categories": {}}

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
        print(f"❌ Error reading memory.json: {e}")
        return []

def save_chat_history(chats):
    try:
        with open(CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=4)
            f.flush()
            os.fsync(f.fileno())  # Forces the OS to physically write data to the laptop storage instantly
    except Exception as e: 
        print(f"❌ Storage write error: {e}")

load_data()

# ------------------ SMART SEMANTIC TOKEN ENGINE ------------------

def get_clean_words(text):
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    
    noise_words = {
        "what", "is", "are", "the", "in", "language", "course", "of", "please", 
        "show", "me", "explain", "about", "tell", "you", "who", "an", "to"
    }
    
    cleaned_set = set()
    for w in words:
        if w in noise_words:
            continue
        if w.endswith("ies"): w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss"): w = w[:-1]
        elif w.endswith("ing"): w = w[:-3]
        elif w == "func": w = "function"
        elif w == "var": w = "variable"
            
        if len(w) >= 1:
            cleaned_set.add(w)
    return cleaned_set

def deep_search_curriculum(user_query):
    query_words = get_clean_words(user_query)
    if not query_words:
        return None

    target_lang = None
    if "python" in query_words: target_lang = "python"
    elif "java" in query_words: target_lang = "java"
    elif "c" in query_words: target_lang = "c"

    concept_aliases = {
        "loop": {"loop", "iteration", "for", "while", "control"},
        "variable": {"variable", "scope", "memory", "type", "allocate"},
        "function": {"function", "prototype", "modular", "logic", "def"},
        "tuple": {"tuple", "sequence", "immutable"},
        "list": {"list", "sequence", "collection", "append"},
        "operator": {"operator", "arithmetic", "math", "logic"}
    }

    expanded_query = set(query_words)
    for core_concept, structural_keywords in concept_aliases.items():
        if core_concept in query_words:
            expanded_query.update(structural_keywords)

    best_topic = None
    highest_score = 0
    matched_cat = ""
    matched_lvl = ""

    for cat_name, cat_data in categ.items():
        if not isinstance(cat_data, dict):
            continue
        lang_prefix = cat_name.lower().split()[0]
        lang_multiplier = 1.0
        
        if target_lang:
            if target_lang == lang_prefix:
                lang_multiplier = 3.0  
            else:
                lang_multiplier = 0.05 

        for level_name, level_data in cat_data.items():
            if not isinstance(level_data, dict) or "topics" not in level_data:
                continue
                
            for topic in level_data["topics"]:
                title_words = get_clean_words(topic.get("title", ""))
                definition_words = get_clean_words(topic.get("definition", ""))
                
                title_match_count = len(expanded_query.intersection(title_words))
                def_match_count = len(expanded_query.intersection(definition_words))
                score = (title_match_count * 25) + (def_match_count * 5)
                
                if user_query.lower().strip() == topic.get("title", "").lower().strip():
                    score += 100
                
                final_score = score * lang_multiplier
                if final_score > highest_score and score > 0:
                    highest_score = final_score
                    best_topic = topic
                    matched_cat = cat_name
                    matched_lvl = level_name

    if highest_score < 15:
        for cat_name, cat_data in categ.items():
            lang_prefix = cat_name.lower().split()[0]
            if lang_prefix in query_words and len(query_words) <= 2:
                response = f"# 🚀 Category: {cat_name.title()}\n\n"
                if "description" in cat_data:
                    response += f"**Description:** {cat_data['description']}\n\n"
                if "history" in cat_data:
                    response += f"**History:** {cat_data['history']}\n\n"
                return response.strip()

    if best_topic and highest_score >= 10:
        title = best_topic.get("title", "").title()
        definition = best_topic.get("definition", "")
        explanation = best_topic.get("explanation", "")
        
        response = f"# 🚀 Topic Found: {title}\n"
        response += f"**Category:** {matched_cat.title()} ({matched_lvl.title()})\n\n"
        response += f"**Definition:** {definition}\n\n"
        if explanation:
            response += f"### 📘 Detailed Breakdown:\n{explanation}\n\n"
            
        examples = best_topic.get("code_examples", [])
        if examples:
            response += "### 💻 Code Examples:\n"
            for index, ex in enumerate(examples, 1):
                response += f"```javascript\n// Example #{index}\n{ex}\n```\n\n"
        return response.strip()

    return None

def get_bot_response(user_message):
    load_data()
    clean_msg = user_message.lower().strip()
    
    for intent in intents:
        if not intent.get("patterns"):
            continue
        for pattern in intent.get("patterns", []):
            if pattern.lower() == clean_msg:
                return random.choice(intent["responses"])

    curriculum_match = deep_search_curriculum(user_message)
    if curriculum_match:
        return curriculum_match

    for intent in intents:
        if not intent.get("patterns"):
            continue
        for pattern in intent.get("patterns", []):
            if len(pattern) > 3 and pattern.lower() in clean_msg:
                return random.choice(intent["responses"])

    return "🤖 I'm sorry, I couldn't find a precise match for that topic in my local documentation. Try asking specifically like 'loops in python', 'variables and scope', or type 'help me'."

# ------------------ ROUTING ENDPOINTS ------------------

@CHATBOT.route("/")
def home():
    return render_template("index.html")

@CHATBOT.route("/get", methods=["POST"])
def chatbot_api():
    try:
        if not request.is_json:
            return jsonify({"response": "❌ Request must be JSON"}), 400
            
        user_message = request.json.get("message", "").strip()
        chat_name = request.json.get("chat_name", "").strip()

        if not user_message:
            return jsonify({"response": "Please type a message 📝"})

        bot_response = get_bot_response(user_message)
        chats = load_chat_history()
        
        # Smart Naming: Extracts title out of the user's first original question prompt
        final_chat_name = chat_name
        if chat_name.startswith("New Chat Thread"):
            words = user_message.split()
            final_chat_name = " ".join(words[:4])
            if len(words) > 4:
                final_chat_name += "..."
            
            existing_names = [c.get("name") for c in chats]
            if final_chat_name in existing_names:
                final_chat_name += f" ({random.randint(10,99)})"

        # Update the active file tracking sequence
        chat_found = False
        for chat in chats:
            if chat.get("name") == chat_name:
                chat["name"] = final_chat_name
                if "messages" not in chat: chat["messages"] = []
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
        return jsonify({"response": f"❌ Server Error: {str(e)}"})

@CHATBOT.route("/get_chats", methods=["GET"])
def get_chats(): 
    return jsonify(load_chat_history())

@CHATBOT.route("/save_chat", methods=["POST"])
def save_chat():
    try:
        req_data = request.get_json()
        chat_name = req_data.get("name")
        messages = req_data.get("messages", [])
        
        if not chat_name:
            return jsonify({"status": "error", "message": "Missing thread name"}), 400

        chats = load_chat_history()
        for chat in chats:
            if chat.get("name") == chat_name:
                chat["messages"] = messages
                save_chat_history(chats)
                return jsonify({"status": "saved"})
                
        chats.append({"name": chat_name, "messages": messages})
        save_chat_history(chats)
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@CHATBOT.route("/delete-chat", methods=["POST"])
def delete_chat():
    try:
        req_data = request.get_json()
        chat_name = req_data.get("chat_name", "").strip()
        
        if not chat_name:
            return jsonify({"success": False, "message": "Missing chat name parameter"}), 400
            
        chats = load_chat_history()
        updated_chats = [c for c in chats if c.get("name").strip() != chat_name]
        
        save_chat_history(updated_chats)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    CHATBOT.run(host="0.0.0.0", port=port, debug=False)