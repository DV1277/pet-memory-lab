from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="memories")

facts = ["My favorite drink is matcha.",
        "I have a dog named Luna.",
        "I go hiking on weekends."
        ]
fact_embeddings = model.encode(facts)
collection.upsert(ids=["1", "2", "3"],
               documents = facts,
               embeddings=fact_embeddings.tolist()
               )

question = input("Ask a question about memories?: ")
question_embedding = model.encode(question)

results = collection.query(query_embeddings = [question_embedding.tolist()], n_results = 1)
print("Most relevant memory: ", results['documents'][0][0])
print("Memories stored: ", collection.count())
print("Question embedding shape: ", question_embedding.shape)