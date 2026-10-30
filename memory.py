from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

facts = ["My favorite drink is matcha.",
        "I have a dog named Luna.",
        "I go hiking on weekends."
        ]
fact_embeddings = model.encode(facts)

question = "What is my pet's name?"
question_embedding = model.encode(question)

highestSimilarity = -1
bestFact = None
for fact, fact_embedding in zip(facts, fact_embeddings):
    similarity = util.cos_sim(fact_embedding, question_embedding)
    if (similarity.item() >= highestSimilarity):
        highestSimilarity = similarity.item()
        bestFact = fact
    print(f"{similarity.item():.3f} {fact}")
print("Best memory: ", bestFact)
print("Score: ", highestSimilarity)

print("Question embedding shape: ", question_embedding.shape)