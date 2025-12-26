import math
from collections import defaultdict

# ========================
# Tokenizer
# ========================
def tokenize(text):
    """
    Simple tokenizer: lowercase and split by spaces
    """
    return text.lower().split()

# ========================
# Indexer
# ========================
class Indexer:
    def __init__(self, documents):
        self.documents = documents
        self.num_docs = len(documents)
        self.tf = []             # Term Frequency per document
        self.df = defaultdict(int)  # Document Frequency per term
        self.build_index()

    def build_index(self):
        """
        Build TF and DF for all documents
        """
        for doc in self.documents:
            term_freq = defaultdict(int)
            terms = tokenize(doc)

            for term in terms:
                term_freq[term] += 1

            self.tf.append(term_freq)

            for term in term_freq:
                self.df[term] += 1

# ========================
# Scoring Models
# ========================
class ScoringModel:
    """
    Interface for scoring models
    """
    def score(self, doc_tf, query_vector):
        raise NotImplementedError

class TFIDFModel(ScoringModel):
    def __init__(self, df, num_docs):
        self.df = df
        self.num_docs = num_docs

    def tfidf_weight(self, tf, term):
        """
        Compute TF-IDF weight for a term
        """
        idf = math.log(self.num_docs / self.df[term])
        return tf * idf

    def score(self, doc_tf, query_vector):
        """
        Compute score of a document for the given query
        """
        score = 0.0
        for term, q_weight in query_vector.items():
            if term in doc_tf:
                score += self.tfidf_weight(doc_tf[term], term) * q_weight
        return score

# ========================
# Query Processor
# ========================
class QueryProcessor:
    def __init__(self, df, num_docs):
        self.df = df
        self.num_docs = num_docs

    def process(self, query):
        """
        Convert query into a weighted vector
        """
        terms = tokenize(query)
        q_vector = defaultdict(int)

        for term in terms:
            if term in self.df:
                q_vector[term] += 1

        for term in q_vector:
            q_vector[term] *= math.log(self.num_docs / self.df[term])

        return q_vector

# ========================
# Search Engine
# ========================
class SearchEngine:
    def __init__(self, indexer, scoring_model):
        self.indexer = indexer
        self.model = scoring_model

    def search(self, query_vector):
        """
        Score and rank all documents
        """
        scores = []

        for doc_id, doc_tf in enumerate(self.indexer.tf):
            score = self.model.score(doc_tf, query_vector)
            scores.append((doc_id, score))

        # Sort by descending score
        return sorted(scores, key=lambda x: x[1], reverse=True)

# ========================
# Main Execution
# ========================
def main():
    # Load documents
    with open("documents.txt", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    # Build index
    indexer = Indexer(documents)

    # Initialize TF-IDF model and query processor
    scoring_model = TFIDFModel(indexer.df, indexer.num_docs)
    query_processor = QueryProcessor(indexer.df, indexer.num_docs)

    # Initialize search engine
    engine = SearchEngine(indexer, scoring_model)

    # Get user query
    query = input("Enter your query: ")
    query_vector = query_processor.process(query)

    # Search and display results
    results = engine.search(query_vector)

    print("\nSearch Results:")
    for doc_id, score in results:
        if score > 0:
            print(f"Doc {doc_id + 1} | Score: {score:.3f}")

if __name__ == "__main__":
    main()
