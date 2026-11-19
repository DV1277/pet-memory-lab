from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from pathlib import Path
import numpy as np
import csv
import warnings
import re

model = SentenceTransformer("all-MiniLM-L6-v2")
project_dir = Path(__file__).resolve().parent
client = chromadb.PersistentClient(path=str(project_dir/ "chroma_data"))
collection = client.get_or_create_collection(name="memories")


def add_fact(fact, *, subject=None, detail=None):
    if not isinstance(fact, str) or not fact.strip():
        raise ValueError("Fact must not be blank.")

    options = {}
    if subject is not None or detail is not None:
        options["metadatas"] = [memory_key(subject, detail)]

    fact_embedding = model.encode(fact)
    fact_id = str(uuid.uuid4())
    collection.add(ids = [fact_id],
               documents = [fact],
               embeddings = [fact_embedding.tolist()],
               **options,
               )
    sync_facts_csv()
    return {"id": fact_id, "fact": fact}

def memory_key(subject, detail):
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (subject, detail)
    ):
        raise ValueError("Subject and detail must be nonblank text.")

    return {
        "subject": " ".join(subject.casefold().split()),
        "detail": " ".join(detail.casefold().split()),
    }


def lookup_fact(subject, detail):
    key = memory_key(subject, detail)
    result = collection.get(
        where={
            "$and": [
                {"subject": key["subject"]},
                {"detail": key["detail"]},
            ]
        },
        include=["documents"],
        limit=2,
    )

    if not result["ids"]:
        return None
    if len(result["ids"]) > 1:
        raise ValueError("Multiple facts match this subject and detail.")

    return {"id": result["ids"][0], "fact": result["documents"][0]}

def list_facts():
    fact_records = collection.get()
    memories = []
    for fact_id, fact in zip(fact_records["ids"], fact_records["documents"]):
        memories.append({"id": fact_id, "fact": fact})
    return memories


def search_facts(question):
    """Return the nearest stored fact and id, none when collection is currently empty"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be blank.")

    if not any(char.isalnum() for char in question):
        raise ValueError("Question must contain a letter or number.")
    if (collection.count() == 0):
        return None

    question_embedding = model.encode(question)
    results = collection.query(query_embeddings = [question_embedding.tolist()], n_results = min(2, collection.count()))

    candidates = []
    for fact_id, fact, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0]
    ):
        candidates.append({
            "id": fact_id,
            "fact": fact,
            "distance": distance,
        })

    if not candidates:
        return None
    rerank_texts = ["Question: " + question]
    rerank_texts += ["Answer: " + candidate["fact"] for candidate in candidates]
    rerank_embeddings = model.encode(rerank_texts)
    for index, candidate in enumerate(candidates):
        candidate["rerank_distance"] = float(np.sum((rerank_embeddings[0] - rerank_embeddings[index+1]) **2))
    candidates.sort(key=lambda candidate: candidate["rerank_distance"])

    top = candidates[0]

    runner_up = candidates[1] if len(candidates) > 1 else None
    return {"id": top["id"],
            "fact": top["fact"],
            "distance": top["distance"],
            "runner_up": runner_up,
            "distance_gap": (
                runner_up["rerank_distance"] - top["rerank_distance"]
                if runner_up is not None else None
            ),
        }

def update_fact(fact_id, new_fact):
    """Update a stored fact by ID, or return None if the ID doesn't exist."""
    if not isinstance(new_fact, str) or not new_fact.strip():
            raise ValueError("Fact must not be blank.")
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