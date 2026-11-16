# Pet Memory Lab 🐾

Fall 2026 AI Collective at UCSD Project: give a pet companion memories about its owner, ask a question, and compare what **keyword search** and **embedding search** find.

We’re starting small and can build on it later. The first version is about getting memory retrieval working and comparing the results. Fancy pet features can wait.

## What we’re building

1. **Add a fact**, like “My favorite drink is matcha.”
2. **Ask a question**, like “What do I like to drink?”
3. **Show both results side by side** using the same facts and questions.

Keyword search matches words. Embedding search looks for similar meaning. For now, the results can be the retrieved memories themselves; we don’t need a paid AI API.

## Week 5

**Everything is due Friday, October 30, 2026.**

- **Kavya:** a runnable skeleton interface with Add a fact, Ask a question, and two result areas. Clearly labeled mock results are fine this week.
- **Carlos:** basic embedding retrieval running with sample facts, storage, a working query, and setup instructions.
- **Kenny:** catch up with DV and pick a small contribution with Kavya or Carlos. The exact piece will be agreed after the briefing.
- **DV:** share the repo and starter setup, brief Kenny, and help agree on how the interface and search code connect.

See [the checklist](CHECKLIST.md) for the small steps. Put your code and run instructions here and share a short update by Friday. If you’re stuck or need more time, tell the group early so we can help.

## Proposed stack

We’ll confirm this before connecting everything:

| Piece | Starting point |
| --- | --- |
| Main code | Python |
| Interface | [Streamlit](https://docs.streamlit.io/) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) with `all-MiniLM-L6-v2` |
| Vector database | [Chroma](https://docs.trychroma.com/), stored locally |
| Keyword baseline | Simple Python word matching |
| Checks | pytest |
| Dependencies | `requirements.txt` and a virtual environment |

## Getting started

Clone the repo, create a virtual environment, and install the dependencies below. Then run the retrieval example or the Streamlit dashboard.

```bash
git clone https://github.com/DV1277/pet-memory-lab.git
cd pet-memory-lab
python3 -m venv .venv
source .venv/bin/activate
```
### Install dependencies

After creating and activating your virtual environment:

1. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

All files downloaded when embedding model loads on first use. Memories are store locally in chroma in 'chroma_data/' folder. See **Using memory backend** section below for available function examples.

Running `python memory.py` still seeds those sample facts, asks one question, and prints the nearest memory. Importing `memory.py` does not prompt for input.

### Run the dashboard

After the install step above:

```bash
streamlit run dashboard.py
```

The dashboard saves facts, lists them with ids, and searches the same stored memories with keyword and embedding results side by side. It calls three functions in `memory.py`:

- `add_fact(fact)` saves one fact and returns `{"id": ..., "fact": ...}`.
- `list_facts()` returns each saved fact as `{"id": ..., "fact": ...}`.
- `search_facts(question)` returns the nearest fact as `{"id": ..., "fact": ...}`, or `None` when nothing is saved.

Keyword search lives in `dashboard.py`. It ranks the current `list_facts()` output by overlapping words and does not write to Chroma.

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.


## Using the memory backend

Facts are also saved in 'memories.csv' in the project folder. The CSV refreshes when a fact is added, updated, and after migrated IDs in the main collection. Chroma is still used to store embeddings and search memories.

Import the backend in your Python file
```python
import memory
```

### Adding a fact
Function takes in a fact as text, creates embedding using the model, creates a unique id for the fact, and then saves all three parts in Chroma.

Example call:
```Python
new_fact1 = memory.add_fact("Twin peaks is my favorite TV show")
print(new_fact1)
```

The fact is then saved in Chroma and returns a dictionary of the new fact, specifically providing the fact id and fact itself. Each call adds a new record

### Getting a list of facts
Function takes no arguments. Returns a list of dictionaries, memories, each containing the fact and its unique ID.

Example call:
```Python
facts = memory.list_facts()
print(facts)
```

### Searching for a fact
Function takes in question as an argument. Checks for empty Chroma collection, returns None if there are no facts. It embeds the question and searches for nearest  memory in Chroma if collection isn't empty. Returns dictionary containing that fact, the fact's unique ID, and its distance. Smaller distance means closer match. Nearest result isn't guaranteed to be relevant yet.

Example call:
```python
matchingFact = memory.search_facts("What is my favorite tv show")
print(matchingFact)
```

### Updating a fact

The function takes in the ID of an existing fact and the new fact text. It
updates the fact and its embedding while keeping the same ID. It returns a dictionary with the ID and updated fact, or `None` if the ID doesn't exist.

Example call:
```python
updatedFact = memory.update_fact(fact_id, "My pet is named Teo.")
print(updatedFact)
```

`fact_id` should be the ID of the fact you want to update.

### Updating old fact IDs

Older saved facts might have numeric IDs like `1`, `2`, and `3`. This migration
changes numeric IDs to UUIDs and keeps the fact text, embedding, and metadata.
Facts that already have UUIDs won't be changed.

Before running it, stop the dashboard and make a backup of your `chroma_data/`
folder. Then run this command from the project folder:

```bash
python -c "import memory; print(memory.migrate_numeric_ids(memory.collection))"
```

The output shows the old and new IDs. If there are no numeric IDs left, it
returns `{}`. Any tests or other references using the old IDs will need to be
updated.

## After Friday

Connect the interface to real retrieval, choose who owns the keyword baseline, and compare both methods on the same facts and questions.

- **Week of November 16:** virtual project check-in. Show progress, explain who this helps, and share what’s next.
- **December 3:** project showcase. Present the working demo and what we learned.

## Meeting notes

The [edited first meeting recap](docs/Meeting-1-Recap.pdf) is our reference for the October 27 decisions and this week’s tasks. It replaces the earlier draft recap.
