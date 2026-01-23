import os
from dotenv import load_dotenv

# loading envs from .env file
load_dotenv()

class Settings:
    def __init__(self):
        # google api key for gemini
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # pinecone config
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "veritas-financial-index")
        
        # --- neo4j graph database config ---
        raw_uri = os.getenv("NEO4J_URI", "")
        
        # AUTO-FIX: removing trailing port if user added it
        # because it causes dns errors with some drivers
        if ":7687" in raw_uri:
            raw_uri = raw_uri.replace(":7687", "")
            
        # removing extra spaces just in case
        self.NEO4J_URI = raw_uri.strip()
        
        self.NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        
        self._validate()

    def _validate(self):
        missing = []
        
        # checking google
        if not self.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
            
        # checking pinecone
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
            
        # checking neo4j 
        if not self.NEO4J_URI:
            missing.append("NEO4J_URI")
        if not self.NEO4J_USERNAME:
            missing.append("NEO4J_USERNAME")
        if not self.NEO4J_PASSWORD:
            missing.append("NEO4J_PASSWORD")
        
        if missing:
            raise ValueError(f"critical config error: missing environment variables: {', '.join(missing)}")

# initializing the settings object
settings = Settings()