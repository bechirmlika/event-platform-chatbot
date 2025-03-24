import json
import numpy as np
from config.db import get_db_connection
from config.llm import get_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem.porter import PorterStemmer
import nltk

def preprocess_text(text, language='english'):
    """Preprocess text to improve search."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Stopwords removal (optional)
    # To activate, uncomment and download stopwords with nltk.download('stopwords')
    # from nltk.corpus import stopwords
    # stop_words = set(stopwords.words(language))
    # tokens = word_tokenize(text, language=language)
    # text = ' '.join([word for word in tokens if word not in stop_words])
    
    # Stemming (reducing words to their root)
    stemmer = PorterStemmer()  # Using English Porter Stemmer instead of French
    tokens = nltk.word_tokenize(text, language=language)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def load_faq():
    """Load FAQ questions from a JSON file."""
    with open("data/faqData.json", "r", encoding="utf-8") as f:
        return json.load(f)

def search_faq(user_message, threshold=0.6):
    """
    Enhanced FAQ search using:
    - Text preprocessing
    - Cosine similarity with adjustable threshold
    - Keyword matching
    """
    faqs = load_faq()
    
    # Preprocess user message
    processed_user_message = preprocess_text(user_message)
    
    # Initialize variables for best result
    best_match = None
    best_score = 0
    
    # Create TF-IDF vectorizer
    all_texts = [processed_user_message]
    
    # Preprocess FAQ questions
    processed_questions = []
    for faq in faqs:
        processed_question = preprocess_text(faq["question"])
        processed_questions.append(processed_question)
        all_texts.append(processed_question)
    
    # Calculate similarity scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compare with each FAQ question
    for i, faq in enumerate(faqs):
        # Cosine similarity measure
        cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[i+1:i+2])[0][0]
        
        # Additional keyword check: verify if important words are shared
        user_words = set(processed_user_message.split())
        faq_words = set(processed_questions[i].split())
        shared_words = user_words.intersection(faq_words)
        
        # Calculate composite score (cosine + bonus for shared words)
        word_match_bonus = len(shared_words) / max(len(user_words), 1) * 0.2
        total_score = min(cosine_score + word_match_bonus, 1.0)
        
        # Keep the best match
        if total_score > best_score:
            best_score = total_score
            best_match = faq["answer"]
        
        # Debug - display in logs
        print(f"FAQ Match: {faq['question']} - Score: {total_score}")
    
    # Return response if score is above threshold
    if best_score >= threshold:
        return best_match
    else:
        # Debug - log when no match is found
        print(f"No FAQ match found. Best score: {best_score}, Threshold: {threshold}")
        return None

def search_database(message):
    """Searches MySQL for events, speakers, sponsors, etc., based on intent."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    message_lower = message.lower()
    results = {}
    
    # Si la requête concerne un événement spécifique
    if "événement" in message_lower or "event" in message_lower:
        # Extrait le nom potentiel de l'événement de la requête
        potential_event_names = extract_entity_names(message)
        
        if potential_event_names:
            for event_name in potential_event_names:
                # Recherche l'événement spécifique
                cursor.execute("SELECT id, title, category, description, startDate FROM events WHERE title LIKE %s", 
                              (f"%{event_name}%",))
                event_results = cursor.fetchall()
                
                if event_results:
                    results['events'] = event_results
                    
                    # Pour chaque événement trouvé, recherche les speakers associés
                    for event in event_results:
                        cursor.execute("""
                            SELECT s.firstName, s.lastName, s.company 
                            FROM components_user_speakers s
                            JOIN event_speakers es ON s.id = es.speaker_id
                            WHERE es.event_id = %s
                        """, (event['id'],))
                        event_speakers = cursor.fetchall()
                        
                        if event_speakers:
                            results['event_speakers'] = event_speakers
                    
                    # Recherche les sponsors associés à l'événement
                    for event in event_results:
                        cursor.execute("""
                            SELECT sp.name, sp.email 
                            FROM sponsors sp
                            JOIN event_sponsors es ON sp.id = es.sponsor_id
                            WHERE es.event_id = %s
                        """, (event['id'],))
                        event_sponsors = cursor.fetchall()
                        
                        if event_sponsors:
                            results['event_sponsors'] = event_sponsors
    
    # Code original pour la recherche générale
    if "speaker" in message_lower:
        query = "SELECT firstName, lastName, company FROM components_user_speakers"
        cursor.execute(query)
        results['speakers'] = cursor.fetchall()
    
    if "event" in message_lower and not results.get('events'):
        query = "SELECT title, category, description, startDate FROM events"
        cursor.execute(query)
        results['events'] = cursor.fetchall()
    
    if "sponsor" in message_lower and not results.get('event_sponsors'):
        query = "SELECT name, email FROM sponsors"
        cursor.execute(query)
        results['sponsors'] = cursor.fetchall()
    
    # Recherche générique si aucun résultat spécifique n'est trouvé
    if not results:
        terms = message.split()
        for term in terms:
            search_term = f"%{term}%"
            
            # Recherches dans les différentes tables
            # [Code original de recherche]
    
    conn.close()
    return results

def extract_entity_names(message):
    """Extrait les noms potentiels d'entités (événements, personnes) d'un message."""
    # Liste de mots à ignorer dans l'extraction
    stop_words = ["le", "la", "les", "de", "du", "des", "un", "une", 
                  "et", "ou", "pour", "à", "au", "aux", "dans", "par", 
                  "sur", "avec", "sans", "speaker", "speakers", "événement", 
                  "event", "sponsor", "sponsors", "donne", "recommande", "plateforme"]
    
    # Convertit le message en minuscules et divise en mots
    words = message.lower().split()
    
    # Trouve les séquences de mots qui ne sont pas des stop words
    entities = []
    current_entity = []
    
    for word in words:
        cleaned_word = word.strip(",.!?;:\"'()[]{}").lower()
        if cleaned_word and cleaned_word not in stop_words:
            current_entity.append(word)
        elif current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
    
    # Ajoute la dernière entité si elle existe
    if current_entity:
        entities.append(" ".join(current_entity))
    
    # Filtre les entités trop courtes
    entities = [e for e in entities if len(e) > 2]
    
    return entities

def generate_llm_response(message, db_results):
    """Utilise Gemini AI pour générer une réponse concise et structurée."""
    llm = get_llm()
    
    # Instructions pour des réponses plus concises et structurées
    instructions = """
    Format your response following these guidelines:
    1. Keep your answer concise (max 200 words)
    2. Use bullet points for lists
    3. Focus on 3-5 key recommendations only
    4. Include name, role, and company for each person
    5. For event-specific queries, clearly state the relationship between speakers/sponsors and events
    """
    
    context = f"Voici les résultats pertinents de la base de données : {db_results}"
    full_message = f"{instructions}\n\n{message}\n\nContexte:\n{context}"
    
    return llm.predict(full_message)


def filter_irrelevant_questions(user_message):
    """Filters off-topic questions and returns a standard response."""
    # Extended list of irrelevant keywords for an event platform
    irrelevant_keywords = [
        # Commerce and products
        "price", "offer", "customer service", "policy", "opening hours",
        "iphone", "apple", "samsung", "phone", "smartphone", "mobile",
        "product", "buy", "purchase", "sale", "store", "shop", "retail",
        "order", "shipping", "delivery", "refund", "warranty", "discount",
        
        # Technology (non-event related)
        "computer", "laptop", "mouse", "keyboard", "printer", "scanner",
        "windows", "macos", "android", "ios", "application", "software", "app",
        "download", "install", "update", "bug", "virus", "malware",
        "tech support", "troubleshooting", "hardware", "processor", "ram",
        
        # Food and restaurants
        "restaurant", "cafe", "food", "menu", "meal", "cuisine",
        "recipe", "ingredient", "dish", "beverage", "lunch", "dinner",
        "breakfast", "coffee", "dessert", "snack", "catering",
        
        # Transportation and travel (non-event related)
        "flight", "plane", "train", "bus", "taxi", "uber", "hotel",
        "booking", "travel", "destination", "visa", "passport", "luggage",
        "vacation", "holiday", "trip", "journey", "airport", "station",
        
        # Health and medicine
        "doctor", "hospital", "medicine", "pharmacy", "treatment",
        "symptom", "disease", "appointment", "consultation", "diagnosis",
        "health", "clinic", "medical", "therapy", "healing", "surgery",
        
        # Finance and banking
        "bank", "account", "credit", "card", "loan", "mortgage", "investment",
        "stock market", "share", "rate", "interest", "insurance", "finance",
        "banking", "transaction", "transfer", "deposit", "withdrawal", "atm",
        
        # Real estate
        "apartment", "house", "rent", "lease", "real estate purchase",
        "property", "real estate agent", "mortgage", "tenant", "landlord",
        "housing", "condo", "townhouse", "deed", "ownership", "listing",
        
        # General education (not event-related)
        "school", "university", "degree", "course", "student", "professor",
        "exam", "grade", "homework", "study", "training", "academic",
        "college", "curriculum", "classroom", "textbook", "assignment",
        
        # Entertainment (non-event related)
        "movie", "series", "video game", "streaming", "netflix", "spotify",
        "music", "album", "song", "artist", "actor", "director", "theater",
        "show", "concert", "performance", "play", "cinema", "tv show",
        
        # Social media and personal communication
        "facebook", "instagram", "twitter", "tiktok", "snapchat", "youtube",
        "message", "email", "call", "contact", "discuss", "chat", "text",
        "whatsapp", "telegram", "signal", "dm", "friend request", "follower",
        
        # Vague words without event context
        "best", "worst", "how", "why", "when", "where", "who",
        "help", "problem", "solution", "need", "urgent", "quick", "fast",
        "easy", "difficult", "simple", "complex", "good", "bad", "better"
    ]
    
    # Filter for questions directly related to events
    event_related_keywords = [
        "event", "conference", "webinar", "workshop", "forum", "summit", 
        "meetup", "hackathon", "session", "speaker", "presenter", "program", 
        "agenda", "schedule", "calendar", "event planning", "organize", 
        "participant", "audience", "networking", "sponsor", "partner",
        "badge", "registration", "register", "ticket", "entry", "admission",
        "evey", "exhibitor", "exhibit", "stand", "booth", "presentation",
        "panel", "discussion", "debate", "keynote", "speech", "talk",
        "guest", "VIP", "venue", "location", "hall", "date", "time",
        "attendee", "delegate", "moderator", "host", "facilitator", "planner",
        "event management", "livestream", "virtual event", "hybrid event",
        "in-person", "online", "check-in", "RSVP", "invite", "invitation"
    ]
    
    user_message_lower = user_message.lower()
    
    # Check if message contains irrelevant keywords
    has_irrelevant = any(keyword in user_message_lower for keyword in irrelevant_keywords)
    
    # Check if message contains at least one event-related keyword
    has_event_related = any(keyword in user_message_lower for keyword in event_related_keywords)
    
    # If message contains irrelevant keywords and doesn't contain event-related keywords
    if has_irrelevant and not has_event_related:
        return "Sorry, this question is not supported by our events platform. Evey.live specializes in organizing and managing professional events."
    
    return None

def get_best_response(user_message):
    """Pipeline optimisé pour la recherche FAQ, MySQL et LLM."""
    
    # 1. Vérification des questions hors sujet
    irrelevant_response = filter_irrelevant_questions(user_message)
    if irrelevant_response:
        return irrelevant_response
    
    # 2. Vérification des questions FAQ
    faq_response = search_faq(user_message)
    if faq_response:
        return faq_response
    
    # 3. Recherche MySQL (événements, speakers, sponsors)
    db_results = search_database(user_message)
    if any(db_results.values()):
        return generate_llm_response(user_message, db_results)
    
    # 4. Si aucune correspondance n'est trouvée, utiliser Gemini avec contexte vide
    return generate_llm_response(user_message, "Aucune donnée pertinente trouvée.")