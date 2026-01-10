# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Book recommendation system using the Hardcover API (hardcover.app). Collects user/book data, builds recommendations using matrix factorization (TensorFlow), and provides a Flask web app for finding "book friends" with similar reading tastes.

## Quick Start Commands

```bash
# Setup
mkdir -p ~/data/hardcover
pip install requests python-dotenv tensorflow numpy scikit-learn flask gunicorn

# Data collection (25 users per batch, ~30 sec each due to rate limiting)
python3 scripts/process_batch.py

# Download multiple batches
for i in {1..20}; do echo "=== Batch $i/20 ==="; python3 scripts/process_batch.py; sleep 15; done

# Train model and generate webapp data
python3 precompute_all.py

# Run webapp locally
cd webapp && python3 app.py  # http://localhost:8001

# Deploy to Cloud Run
cd webapp && gcloud run deploy book-friend-finder --source . --region us-central1 --allow-unauthenticated
```

## Architecture

### Data Flow

```
Hardcover GraphQL API
         ↓
scripts/process_batch.py (25 users/batch, 1s delay between API calls)
         ↓
~/data/hardcover/
├── users.json         (user profiles)
├── user_books.json    (users → books, ~100MB+)
└── books_users.json   (books → users inverted index, ~60MB+)
         ↓
precompute_all.py (TensorFlow matrix factorization + K-means clustering)
         ↓
webapp/data/
├── recommendations.json (pre-computed friend matches)
├── users.json          (filtered active users)
└── clusters.json       (cluster membership)
         ↓
webapp/app.py (Flask, no ML deps at runtime)
```

### Key Design Decisions

- **Offline ML**: All TensorFlow training happens locally via `precompute_all.py`. The webapp only serves pre-computed JSON - no ML libraries needed in production.
- **Atomic batch processing**: `scripts/process_batch.py` saves progress only after all 4 phases complete (fetch users → fetch books → update inverted index → stats). Safe to interrupt mid-batch.
- **Chat access control**: Users can only message their top 10 matches (enforced in `webapp/app.py:chat_conversation`, lines 211-214).
- **SQLite chat storage**: `webapp/chat_db.py` stores messages in `webapp/data/chat.db`.
- **Progress tracking**: `progress.json` in repo root tracks batches processed. After forced shutdowns, verify all files are consistent (users.json, user_books.json, books_users.json counts should match). If corrupted, `scripts/invert_to_books.py` can rebuild books_users.json from user_books.json.

### Recommendation Algorithm

Matrix factorization with implicit feedback weights:
- Read (rating ≥3 or no rating): 1.0
- Currently reading: 0.7
- Want to read: 0.3
- Did not finish: 0.0

Friend matching uses cosine similarity on L2-normalized user feature vectors, grouped by K-means clusters.

**Tunable constants in `precompute_all.py`:**
- `MIN_RATINGS_PER_USER = 20`, `MIN_USERS_PER_BOOK = 5` (data filtering)
- `NUM_FEATURES = 20` (latent dimensions)
- `NUM_CLUSTERS = 15` (K-means)
- `LAMBDA = 1.0` (L2 regularization)
- `ITERATIONS = 300`, `LEARNING_RATE = 0.1`

## API Configuration

Create `.env`:
```
HARDCOVER_API_TOKEN=your_token_here
```
Get token: https://hardcover.app/account/api

**Book status IDs:** 1=Want to read, 2=Currently reading, 3=Read, 5=Did not finish

## Webapp Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | User selection |
| `/find_friends` | POST | Get friend recommendations |
| `/cluster/<id>` | GET | View cluster members |
| `/chat` | GET | Chat inbox |
| `/chat/<other_id>` | GET/POST | Conversation (top 10 matches only) |
| `/chat/<other_id>/messages` | GET | Poll for new messages (JSON) |
| `/chat/unread_count` | GET | Unread count (JSON) |

## Production

**URL:** https://book-friend-finder-770103525576.us-central1.run.app

See `DEPLOYMENT.md` for full Cloud Run setup guide.

## Legacy Scripts

Do not use `scripts/fetch_users.py`, `fetch_user_books.py`, `append_users.py`, `invert_to_books.py` - superseded by `process_batch.py` (conflicting progress tracking).
