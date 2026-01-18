import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # CHANGED: Now looking for Google
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "veritas-financial-index")
        
        self._validate()

    def _validate(self):
        missing = []
        # CHANGED: Validation logic
        if not self.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        
        if missing:
            raise ValueError(f"CRITICAL CONFIG ERROR: Missing environment variables: {', '.join(missing)}")

settings = Settings()