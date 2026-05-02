
import json
import os

import chromadb


def get_paper_mappings():
    persist_dir = 'c:/Users/Windows/PycharmProjects/NewAIScientist/chroma_db'
    if not os.path.exists(persist_dir):
        print(f"Error: {persist_dir} does not exist")
        return {}

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = client.get_collection('papers')
    except Exception as e:
        print(f"Error: Could not get collection 'papers': {e}")
        return {}

    results = collection.get(include=['metadatas'])
    mappings = {}
    for meta in results.get('metadatas', []):
        p_id = meta.get('paper_id')
        p_title = meta.get('paper_title')
        if p_id and p_title:
            mappings[p_id] = p_title

    return mappings

if __name__ == "__main__":
    mappings = get_paper_mappings()
    with open('paper_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved {len(mappings)} mappings to paper_mappings.json")
