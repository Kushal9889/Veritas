import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neo4j import GraphDatabase
from src.core.config import settings

def check_graph_stats():
    print("🔌 Connecting to Neo4j AuraDB...")
    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"📊 Graph Status:")
            print(f"   Nodes: {node_count}")
            print(f"   Relationships: {rel_count}")
            
            if node_count > 0:
                print("\n🔍 Sampling a few relationships:")
                sample = session.run("MATCH (a)-[r]->(b) RETURN labels(a)[1] as lA, a.id as idA, type(r) as rel, labels(b)[1] as lB, b.id as idB LIMIT 5")
                for record in sample:
                    print(f"   ({record['idA']}:{record['lA']}) -[{record['rel']}]-> ({record['idB']}:{record['lB']})")

    except Exception as e:
        print(f"❌ Failed to query database: {e}")

if __name__ == "__main__":
    check_graph_stats()
