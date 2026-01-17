import os
from dotenv import load_dotenv

# 1. Load the .env file immediately. 
# This looks for .env in the root directory and loads variables into system memory.
load_dotenv()

class Settings:
    def __init__(self):
        # 2. Fetch variables. If "OPENAI_API_KEY" is missing, returns None.
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "veritas-financial-index") # Default value provided
        
        # 3. THE GUARDRAIL (Fail-Fast)
        # We check immediately. If keys are missing, we stop EVERYTHING.
        # This prevents hours of debugging "Authentication Error" later.
        self._validate()

    def _validate(self):
        missing = []
        if not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        
        if missing:
            raise ValueError(f"CRITICAL CONFIG ERROR: Missing environment variables: {', '.join(missing)}")

# 4. Instantiate once. 
# Now other files just import 'settings' and get the guaranteed valid config.
settings = Settings()