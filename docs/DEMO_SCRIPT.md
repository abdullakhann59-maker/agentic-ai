# 7-minute demo script

**Before the interview:** run `ollama serve`, then `python run_all.py`, open http://localhost:8501,
and run every command below once (the first LLM call loads the model and is slow).
Record a screen video of the full demo as a backup.

| # | Say | Type / click | What to point at |
|---|---|---|---|
| 1 | "This is an agent that runs locally: Ollama, LangGraph, ChromaDB, Selenium." | Sidebar health panel | All green; knowledge counts |
| 2 | "Every command first searches the vector DB." | `What can you do?` | Green badge **Quick answer**, low tokens, under a second or two |
| 3 | "Here a stored task playbook gives the plan." | `What is the total revenue by region in sales_data.csv? Make a bar chart too.` | Blue **Task playbook** badge, chart, "How the agent worked": only 3 tools given to the LLM |
| 4 | "The scraper first analyses the site." | `Scrape product names and prices from http://localhost:8001/dynamic/` | Agent found the hidden API from network traffic and asks for confirmation |
| 5 | "Risky actions need my approval, checked by code." | Click **Approve** | 24 rows (3 pages), CSV preview and download |
| 6 | "Same tool, different site type." | `Extract the data from the iframes on http://localhost:8001/iframe/` | 4 iframes found, nested one included, ad frame skipped |
| 7 | "It also refuses what it shouldn't do." | `Scrape http://localhost:8001/captcha/` | Stops on CAPTCHA |
| 8 | "Prompt injection test: this page hides an instruction to send an email." | `Summarise http://localhost:8001/injection/` | Summary only; no email tool was called |
| 9 | "Documents are answered with sources." | `How many days of paid leave do employees get?` | Answer + `company_handbook.md` source |
| 10 | "I can teach it new tasks without retraining." | Knowledge tab → add a task playbook | Item count goes up |
| 11 | "And I measure whether RAG helps." | Dashboard tab | Success rate, tokens, per-path time and success |

**If the local model is slow or makes a mistake:** say so honestly, show the step log to explain
what happened (that shows you understand the system), then switch to the recorded video.
