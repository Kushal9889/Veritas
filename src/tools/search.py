from duckduckgo_search import DDGS

def search_web(query):
    """
    uses duckduckgo to search the real web for free.
    useful for checking if the 10k is outdated.
    """
    print(f"--- 🌐 WEB: searching for '{query}'... ---")
    try:
        # running the search
        results = DDGS().text(query, max_results=3)
        
        # formatting it nicely for the llm
        search_data = ""
        if results:
            for r in results:
                search_data += f"source: {r['title']}\nurl: {r['href']}\ncontent: {r['body']}\n\n"
        else:
            search_data = "no web results found."
            
        return search_data
    except Exception as e:
        print(f"web search failed: {e}")
        return "error searching web."