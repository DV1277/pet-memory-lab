from sentence_transformers import SentenceTransformer
import chromadb
import uuid

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="memories")


def add_fact(fact):
    fact_embedding = model.encode(fact)
    fact_id = str(uuid.uuid4())
    collection.add(ids = [fact_id],
               documents = [fact],
               embeddings = [fact_embedding.tolist()]
               )
    return {"id": fact_id, "fact": fact}

def list_facts():
    fact_records = collection.get()
    memories = []
    for fact_id, fact in zip(fact_records["ids"], fact_records["documents"]):
        memories.append({"id": fact_id, "fact": fact})
    return memories


def search_facts(question):
    """Return the nearest stored fact and id, none when collection is currently empty"""
    # check for empty collection, Kenny should return some sort of error on the frontend 
    if (collection.count() == 0):
        return None
    
    question_embedding = model.encode(question)
    results = collection.query(query_embeddings = [question_embedding.tolist()], n_results = 1)
    return {"id": results["ids"][0][0],
            "fact": results["documents"][0][0]}


# print("Most relevant memory: ", results['documents'][0][0])
# print("Memories stored: ", collection.count())
# print("Question embedding shape: ", question_embedding.shape)
