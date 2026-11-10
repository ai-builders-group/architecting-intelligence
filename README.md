# Architecting Intelligence: companion code

The full code for every listing in the book Architecting Intelligence: Systems Engineering for Production AI Agents and Workflows. Each file is the full listing behind an excerpt in the book; LISTINGS.md maps chapters to files.

Book and updates: arizenai.com/architecting-intelligence/?ref=gh-b2&utm_source=gh-b2#book-signup

## Start here

1. Find your listing. The book points to it as "Full listing lst-N.M in the companion repository". Open `lst/lst-N.M.md` for its landing page, or go straight to the file at `chapters/chNN/lst-N.M.py`.
2. Check its class in `inventory.tsv` (see the table below). The class tells you what the file needs before you run it.
3. Use Python 3.12 or newer. Create a virtual environment and install the packages the file imports. Most listings import `pydantic`; a smaller number import `numpy`, `openai`, `temporalio`, `asyncpg` or `httpx`.
4. Run the file, or import its definitions into your own project.

The code in the book is shown as excerpts. These files are the whole listings behind them. Some files carry marker comments that start with `# --8<--`. They are plain comments and do no harm.

## The four classes

Every listing has one class in `inventory.tsv`.

| Class | Count | What it means |
|---|---|---|
| `runs` | 67 | The file stands alone. It needs only the standard library and ordinary pip packages. |
| `needs-earlier:lst-N.M` | 21 | The file builds on an earlier listing. Put that listing in place first. |
| `needs-service:NAME` | 35 | The file expects a service or an API key, named below. |
| `sketch` | 25 | Partial code, pseudo code, or a configuration shown for reading. It is not meant to be run as it stands. |

## Setup notes for listings that need a service

35 listings expect something outside your machine. The file says where it connects; supply your own account, key or local install.

| Service | Listings | Which |
|---|---|---|
| an LLM API key (OpenAI) | 14 | lst-4.10, lst-5.9, lst-5.12, lst-5.13, lst-6.4, lst-6.11, lst-9.8, lst-9.17, lst-9.22, lst-9.31, lst-9.32, lst-9.34, lst-9.43, lst-9.45 |
| a Temporal server | 6 | lst-6.4, lst-6.5, lst-6.11, lst-6.14, lst-6.15, lst-6.18 |
| PostgreSQL | 4 | lst-5.18, lst-5.19, lst-7.20, lst-9.25 |
| a Hugging Face model download | 3 | lst-1.5, lst-4.5, lst-4.10 |
| Redis | 3 | lst-7.15, lst-9.27, lst-9.37 |
| PostgreSQL with pgvector | 3 | lst-10.3, lst-10.15, lst-11.15 |
| an HTTP API endpoint | 2 | lst-6.4, lst-6.15 |
| a database | 2 | lst-9.31, lst-16.9 |
| an LLM API key (Anthropic) | 1 | lst-5.4 |
| a Slack webhook | 1 | lst-7.17 |
| an SMTP server | 1 | lst-7.18 |
| a message queue | 1 | lst-9.34 |
| a message store | 1 | lst-9.34 |
| Neo4j with Graph Data Science | 1 | lst-10.22 |
| HTTP metrics endpoints | 1 | lst-16.9 |

Nothing in this repository ships credentials. Keep any key you use in your environment and out of your files.

## What is in this repository

- `chapters/chNN/` holds the 148 full listing files, named `lst-N.M`.
- `lst/` holds one landing page per listing.
- `LISTINGS.md` maps each chapter to its listings.
- `inventory.tsv` lists every listing with its class, and `pointers.txt` lists every pointer the book makes into this repository.
- `companion_tooling/` and `docs/TOOLING.md` hold the small offline checks used to keep the listing pages and pointers in order.

## What is checked

The pointer check confirms that all 148 pointers resolve to a file and a landing page. Every Python file in `chapters/` compiles with Python 3.12. The listing pages are generated from `inventory.tsv`.

Before release, the files were also checked for syntax, and every excerpt shown in the book was checked against its full listing: the excerpt's code lines appear in the file, in order.

These checks confirm that the files are present and well formed. They do not exercise the code against live services.

## Errata and questions

Found a bug, or a file that differs from the book? Open an issue in this repository. Fixes land on `main`.

## Licenses

Code is released under the MIT license (`LICENSE`). Diagrams and other non-code content are released under CC BY 4.0 (`LICENSE-CONTENT`). Copyright 2026 AI Builders Group.
