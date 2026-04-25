from Bio import Entrez
import re

Entrez.email = 'test@example.com'

def test_query(q):
    base_query = q + ' AND "free full text"[Filter]'
    # Keep brackets and apostrophes
    safe_query = re.sub(r'[^\w\s\-\(\)\[\]"\'’]', '', base_query)
    print(f"Original: {q}")
    print(f"Safe: {safe_query}")
    handle = Entrez.esearch(db="pubmed", term=safe_query, retmax=5)
    record = Entrez.read(handle)
    handle.close()
    print(f"Count: {record.get('Count', '0')} hits\n")

queries = [
    "Down syndrome AND gut microbiota",
    "Trisomy 21 AND Alzheimer's",
    "chromosome 21 AND microbiota"
]

for q in queries:
    test_query(q)
