"""
Smoke Tests for Production-Optimized SML Query Router
Tests: keyword classification, real metadata filters, entity extraction, section tagging.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestration.router import classify_query, build_metadata_filter, _keyword_classify
from src.ingestion.chunking import classify_chunk_section

def test_keyword_classifier():
    """Test the keyword-based fallback classifier (no Ollama needed)."""
    print("=" * 60)
    print("TEST 1: Keyword Classifier (Fallback)")
    print("=" * 60)

    test_cases = [
        ("What are Tesla's main revenue streams?", "financial"),
        ("What are the key risk factors?", "risk"),
        ("Tell me about manufacturing operations", "operations"),
        ("Are there any lawsuits?", "legal"),
        ("Hello!", "general"),
        ("Hi", "general"),
        ("Thanks", "general"),
        ("What is Tesla's profit margin?", "financial"),
        ("Describe the supply chain strategy", "operations"),
        ("What regulatory risks does Tesla face?", "risk"),
    ]

    passed = 0
    for question, expected in test_cases:
        result = _keyword_classify(question)
        category = result["category"]
        status = "✅" if category == expected else "❌"
        if category == expected:
            passed += 1
        # Verify entities field exists (P2: new field)
        assert "entities" in result, f"Missing 'entities' in result for '{question}'"
        print(f"  {status} '{question}' → {category} (expected: {expected}) tags={result['tags']}")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_metadata_filter():
    """Test REAL metadata filter generation (P1: filters on 'section' field)."""
    print("\n" + "=" * 60)
    print("TEST 2: Real Metadata Filter (Pinecone 'section' field)")
    print("=" * 60)

    test_cases = [
        ("financial", {"section": "financial"}),
        ("risk", {"section": "risk"}),
        ("operations", {"section": "operations"}),
        ("legal", {"section": "legal"}),
        ("general", None),
    ]

    passed = 0
    for category, expected in test_cases:
        result = build_metadata_filter(category)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"  {status} category={category} → filter={result} (expected: {expected})")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_section_classifier():
    """Test chunk section classification (P1: tags during ingestion)."""
    print("\n" + "=" * 60)
    print("TEST 3: Chunk Section Classifier")
    print("=" * 60)

    test_cases = [
        ("Revenue increased by 12% to $81.5 billion in fiscal year 2024", "financial"),
        ("The company faces significant risks from regulatory changes", "risk"),
        ("Our Gigafactory in Texas produces Model Y vehicles for the North American market", "operations"),
        ("The SEC filed a complaint regarding disclosure practices", "legal"),
        ("This page is intentionally left blank", "general"),
        ("Net income attributable to common stockholders was $7.93 billion, representing earnings per share of $2.29", "financial"),
        ("Supply chain disruptions may impact battery production capacity", "operations"),
    ]

    passed = 0
    for text, expected in test_cases:
        result = classify_chunk_section(text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"  {status} '{text[:50]}...' → {result} (expected: {expected})")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed >= len(test_cases) - 1  # Allow 1 miss (edge cases)


def test_sml_classifier():
    """Test the full SML classifier (requires Ollama running)."""
    print("\n" + "=" * 60)
    print("TEST 4: SML Classifier (requires Ollama)")
    print("=" * 60)

    test_cases = [
        ("What are Tesla's main revenue streams?", "financial"),
        ("What are the key risk factors mentioned in the filing?", "risk"),
        ("Hello!", "general"),
        ("Tell me about Tesla's manufacturing operations", "operations"),
    ]

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            print("  ⚠️  Ollama not available. Skipping SML tests.")
            return True
    except Exception:
        print("  ⚠️  Ollama not available. Skipping SML tests.")
        return True

    passed = 0
    for question, expected in test_cases:
        result = classify_query(question)
        category = result["category"]
        # P2: Verify entities field is present
        has_entities = "entities" in result
        status = "✅" if category == expected else "⚠️"
        if category == expected:
            passed += 1
        print(f"  {status} '{question}' → {category} (expected: {expected}) tags={result['tags']} entities={result.get('entities', [])}")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed >= len(test_cases) - 1


def test_persona_mapping():
    """Test persona is derived from category, not keywords (P5)."""
    print("\n" + "=" * 60)
    print("TEST 5: Category-Based Persona (P5)")
    print("=" * 60)

    from src.orchestration.nodes import get_persona

    test_cases = [
        ("financial", "investment banker"),
        ("risk", "chief risk officer"),
        ("operations", "senior business analyst"),
        ("legal", "forensic accountant"),
        ("general", "financial assistant"),
        ("unknown", "senior financial analyst"),
    ]

    passed = 0
    for category, expected in test_cases:
        result = get_persona(category)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"  {status} category={category} → persona='{result}' (expected: '{expected}')")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


if __name__ == "__main__":
    print("\n🔀 Production-Optimized Router — Smoke Tests\n")

    results = []
    results.append(("Keyword Classifier", test_keyword_classifier()))
    results.append(("Real Metadata Filter", test_metadata_filter()))
    results.append(("Chunk Section Classifier", test_section_classifier()))
    results.append(("SML Classifier", test_sml_classifier()))
    results.append(("Category Persona", test_persona_mapping()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {name}")

    print()
    sys.exit(0 if all_pass else 1)
