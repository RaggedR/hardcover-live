# Book Friend Finder

A book recommendation system that uses the [Hardcover](https://hardcover.app) API to find users with similar reading tastes. Uses matrix factorization (TensorFlow) for collaborative filtering and K-means clustering to group readers.

**Live demo:** https://book-friend-finder-770103525576.us-central1.run.app

## How It Works

1. **Data Collection** - Fetches user reading data from Hardcover's GraphQL API
2. **Model Training** - Builds user embeddings via matrix factorization with implicit feedback
3. **Friend Matching** - Finds similar readers using cosine similarity on user vectors
4. **Web App** - Flask app for browsing recommendations and chatting with matches

## Setup

```bash
# Create data directory
mkdir -p ~/data/hardcover

# Install dependencies
pip install requests python-dotenv tensorflow numpy scikit-learn flask gunicorn

# Configure API access
echo "HARDCOVER_API_TOKEN=your_token_here" > .env
```

Get your API token at https://hardcover.app/account/api

## Usage

### Collect Data

```bash
# Fetch one batch (25 users, ~30 sec due to rate limiting)
python3 scripts/process_batch.py

# Fetch multiple batches
for i in {1..20}; do echo "=== Batch $i/20 ==="; python3 scripts/process_batch.py; sleep 15; done
```

### Train Model

```bash
python3 precompute_all.py
```

This generates pre-computed recommendations in `webapp/data/`.

### Run Web App

```bash
cd webapp && python3 app.py
```

Visit http://localhost:8001

## Project Structure

```
hardcover-live/
├── scripts/
│   └── process_batch.py     # Data collection from Hardcover API
├── precompute_all.py        # Model training and recommendation generation
├── webapp/
│   ├── app.py               # Flask application
│   ├── chat_db.py           # SQLite chat storage
│   ├── templates/           # HTML templates
│   └── data/                # Pre-computed recommendations
└── ~/data/hardcover/        # Raw data storage (external)
    ├── users.json
    ├── user_books.json
    └── books_users.json
```

## Algorithm

**Implicit feedback weights:**
- Read (rating >= 3 or no rating): 1.0
- Currently reading: 0.7
- Want to read: 0.3
- Did not finish: 0.0

**Model parameters:**
- 20 latent dimensions
- 15 K-means clusters
- L2 regularization (lambda=1.0)
- 300 training iterations

## Deployment

Deploy to Google Cloud Run:

```bash
cd webapp && gcloud run deploy book-friend-finder \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

See `DEPLOYMENT.md` for full setup guide.

## License

Personal project - not licensed for redistribution.
