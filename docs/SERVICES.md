# Setup notes for listings that need a service

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

Classes: `runs` stands alone with Python packages; `needs-earlier` builds on another listing; `needs-service` expects a service or key; `sketch` is partial code or configuration for reading.
