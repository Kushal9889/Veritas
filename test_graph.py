import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# load envs
load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

def test_connection():
    print("--- 🕸️ TESTING NEO4J CONNECTION ---")
    
    try:
        # connecting to the driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # simple verification query
        driver.verify_connectivity()
        print("✅ connection established successfully.")
        
        # closing driver
        driver.close()
        
    except Exception as e:
        print(f"❌ connection failed: {e}")

if __name__ == "__main__":
    test_connection()