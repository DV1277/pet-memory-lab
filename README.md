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

This repo currently contains the project notes and checklist. App code and dependency versions will be added as we build the first pieces.

```bash
git clone https://github.com/DV1277/pet-memory-lab.git
cd pet-memory-lab
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.

After first code push, add the dependencies to `requirements.txt` and document the exact install and run commands here. Use made-up sample facts for demos.

## After Friday

Connect the interface to real retrieval, choose who owns the keyword baseline, and compare both methods on the same facts and questions.

- **Week of November 16:** virtual project check-in. Show progress, explain who this helps, and share what’s next.
- **December 3:** project showcase. Present the working demo and what we learned.

## Meeting notes

The [edited first meeting recap](docs/Meeting-1-Recap.pdf) is our reference for the October 27 decisions and this week’s tasks. It replaces the earlier draft recap.
