from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from pathlib import Path
import numpy as np
import csv
import warnings

model = SentenceTransformer("all-MiniLM-L6-v2")
project_dir = Path(__file__).resolve().parent
client = chromadb.PersistentClient(path=str(project_dir/ "chroma_data"))
collection = client.get_or_create_collection(name="memories")


def add_fact(fact):
    fact_embedding = model.encode(fact)
    fact_id = str(uuid.uuid4())
    collection.add(ids = [fact_id],
               documents = [fact],
               embeddings = [fact_embedding.tolist()]
               )
    sync_facts_csv()
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
            "fact": results["documents"][0][0],
            "distance": results["distances"][0][0]}

def update_fact(fact_id, new_fact):
    """Update a stored fact by ID, or return None if the ID doesn't exist."""
    existing = collection.get(ids=[fact_id])
    if not existing["ids"]:
        return None
    new_fact_embedding = model.encode(new_fact)
    collection.update(ids=[fact_id],
                      documents = [new_fact],
                      embeddings = [new_fact_embedding.tolist()])
    sync_facts_csv()
    return {"id": fact_id, "fact": new_fact}

def migrate_numeric_ids(target_collection):
    records = target_collection.get(
        include=["documents", "embeddings", "metadatas"]
    )
    migrated = {}

    for index, old_id in enumerate(records["ids"]):
        if not old_id.isdigit():
            continue

        new_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL, f"pet-memory-lab:legacy:{old_id}"
        ))
        fact = records["documents"][index]
        embedding = records["embeddings"][index].tolist()
        metadata = records["metadatas"][index]

        existing = target_collection.get(
            ids=[new_id], include=["documents", "embeddings", "metadatas"]
        )

        if existing["ids"]:
            same_fact = existing["documents"][0] == fact
            same_embedding = np.allclose(existing["embeddings"][0], embedding)
            same_metadata = existing["metadatas"][0] == metadata

            if not (same_fact and same_embedding and same_metadata):
                raise ValueError(f"Replacement ID conflicts with {old_id}")
        else:
            values = {
                "ids": [new_id],
                "documents": [fact],
                "embeddings": [embedding],
            }
            if metadata is not None:
                values["metadatas"] = [metadata]
            target_collection.add(**values)

        replacement = target_collection.get(
        ids=[new_id], include=["documents", "embeddings", "metadatas"]
        )
        if (
            not replacement["ids"]
            or replacement["documents"][0] != fact
            or not np.allclose(replacement["embeddings"][0], embedding)
            or replacement["metadatas"][0] != metadata
        ):
            raise ValueError(f"Could not verify replacement for {old_id}")
        target_collection.delete(ids=[old_id])
        migrated[old_id] = new_id
    if target_collection is collection:
        sync_facts_csv()
    return migrated

def sync_facts_csv():
    facts = list_facts()
    csv_path = project_dir / "memories.csv"
    try:
        with open(csv_path, "w", newline = "", encoding = "utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["id", "fact"])
            writer.writeheader()
            writer.writerows(facts)
    except OSError as error:
        warnings.warn(f"CSV sync failed; Chroma data may still be saved: {error}")