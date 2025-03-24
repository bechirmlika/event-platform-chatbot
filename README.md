# event-platform-chatbot
Chatbot IA Génératif avec LangChain et RAG
- Développement d’un chatbot intelligent intégrant LangChain et la technique Retrieval-Augmented Generation (RAG) pour fournir des réponses basées sur une base de données MySQL.
- Extraction et nettoyage des données textuelles issues de MySQL avant leur transformation en vecteurs numériques
- Utilisation d’un modèle d’embeddings pour la vectorisation des données et stockage des vecteurs dans Pinecone pour une recherche rapide et efficace.
- Recherche des documents pertinents à l’aide de Pinecone et génération de réponses en s’appuyant sur un LLM.
- Support de plusieurs modèles de langage (LLaMA, GPT, Gemini), garantissant une adaptabilité à différents contextes.
- Exposition du chatbot via FastAPI pour permettre une intégration facile dans des applications web et mobiles.
- Technologies : LangChain, RAG, Pinecone, FastAPI, Python, MySQL, LLMs.
