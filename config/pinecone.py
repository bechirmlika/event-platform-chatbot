from pinecone import Pinecone
from config.settings import settings

# Initialisation du client Pinecone avec l'API key
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Récupération des indexes existants
existing_indexes = pc.list_indexes()
print(f"Existing indexes: {existing_indexes}")

# Vérifier si l'index 'evey-db' existe en parcourant la liste des dictionnaires
if any(index_info.get("name") == settings.PINECONE_INDEX for index_info in existing_indexes):
    # Connecter à l'index existant
    index = pc.Index(settings.PINECONE_INDEX)
    
    # Afficher les statistiques de l'index
    print("ℹ️ Pinecone index description:")
    print(index.describe_index_stats())
else:
    print(f"Index {settings.PINECONE_INDEX} does not exist.")
