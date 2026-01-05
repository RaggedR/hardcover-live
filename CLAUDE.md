# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data collection and analysis project for the Hardcover API (hardcover.app), an open-source alternative to Goodreads. The project collects user and book data for building a book recommendation system using collaborative filtering.

**Key components:**
1. **Data collection** - Batch processing of users and books from Hardcover API
2. **Recommendation system** - Hybrid collaborative filtering (8.83% precision)
3. **Friend finding** - K-means clustering to match users with similar tastes
4. **Web application** - Flask app with friend matching and chat features

## Setup

### Dependencies
```bash
# Core dependencies
pip install requests python-dotenv

# For recommendation system
pip install tensorflow numpy scikit-learn

# For webapp
pip install flask gunicorn
```

### Initial Setup
1. Create data directory: `mkdir -p ~/data/hardcover`
2. Create `.env` file with API token (see API Token Setup section)
3. Run first batch: `python3 process_batch.py`

**Important**: The data directory `~/data/hardcover/` must exist before running scripts.

## Data Collection Architecture

### Core Script: `process_batch.py`

The main entry point that orchestrates all data collection. It processes users in batches of 25 and performs **4 atomic phases**:

1. **Phase 1**: Download 25 users from Hardcover API
2. **Phase 2**: Fetch books for each of those 25 users (with 1-second delay between requests)
3. **Phase 3**: Update inverted JSON (books → users mapping) **incrementally**
4. **Phase 4**: Calculate and print statistics

**Critical**: Progress is ONLY updated after ALL 4 phases complete successfully. This ensures data consistency if the script is interrupted.

### Running Data Collection

```bash
# Process next batch of 25 users
python3 process_batch.py

# Process multiple batches (e.g., 10 batches = 250 users)
for i in {1..10}; do
  python3 process_batch.py
  sleep 15  # Pause between batches to avoid rate limits
done
```

### Progress Tracking

Two progress files track state:
- `progress.json` - Overall batch progress (batches_processed, total_users, total_books)
- `book_progress.json` - Legacy file, not used by process_batch.py

**Important**: Never manually edit progress files. They are atomically updated only when all 4 phases complete.

## Data Storage Architecture

### Location
Data files stored in: `~/data/hardcover/` (separate from code)

### Three JSON Files (Updated incrementally)

1. **`users.json`** (~300KB for 1000 users)
   - List of users with profile data
   - Structure: `{metadata: {...}, users: [...]}`

2. **`user_books.json`** (~100MB for 1000 users)
   - Users with their books organized by status
   - Structure: `{metadata: {...}, user_books: [{user: {...}, books: {read: [], currently_reading: [], want_to_read: []}, counts: {...}}]}`

3. **`books_users.json`** (~60MB for 1000 users)
   - **Inverted structure**: Books with their associated users
   - Structure: `{metadata: {...}, books: [{book: {...}, users: [...], user_count: N}]}`
   - Sorted by popularity (user_count descending)

### Incremental Updates (Performance Critical)

Phase 3 uses **incremental merging**, NOT full rebuilding:
- Loads existing `books_users.json`
- Processes ONLY the new 25 users from current batch
- Merges into existing book dictionary
- Re-sorts by popularity

This keeps Phase 3 O(1) time per batch, regardless of total users collected.

## API Details

### Hardcover GraphQL API
- Endpoint: `https://api.hardcover.app/v1/graphql`
- Authentication: Bearer token in `HARDCOVER_API_TOKEN` environment variable
- Rate limit: 60 requests/minute
- Query timeout: 30 seconds max

### API Token Setup
1. Create account at hardcover.app
2. Get token from: https://hardcover.app/account/api
3. Add to `.env` file: `HARDCOVER_API_TOKEN=your_token_here`

### Book Status IDs
- 1 = Want to read
- 2 = Currently reading
- 3 = Read
- 5 = Did not finish

## Statistics Printed

After each batch, the script prints:
- **a) Books with >1 user**: Percentage and count (key metric for recommendation quality)
- **b) Average books per user**: Total book entries / total users
- **c) Number of users processed**: Cumulative total
- **d) Number of books**: Unique books across all users

## Legacy Scripts (Not Used)

These scripts exist but are **superseded by process_batch.py**:
- `fetch_users.py` - Old user fetching (replaced by Phase 1)
- `fetch_user_books.py` - Old book fetching (replaced by Phase 2)
- `append_users.py` - Old incremental user fetching
- `invert_to_books.py` - Old books→users creation (replaced by Phase 3)

**Do not use these scripts** - they have separate progress tracking that conflicts with process_batch.py.

## Key Design Decisions

### Why Batch Size = 25?
- Balances API rate limits (60/min) with reasonable progress increments
- Each batch takes ~30-40 seconds with 1-second delays between user book fetches

### Why Incremental Phase 3?
- At 1000 users, rebuilding from scratch would process all 1000 users every batch
- Incremental approach processes only 25 users per batch, keeping time constant
- Enables scaling to 10,000+ users without performance degradation

### Why Separate Data Directory?
- Keeps large JSON files (100MB+) out of git repository
- Code in `~/git/hardcover/`, data in `~/data/hardcover/`

## Error Handling

### Common Issues

**"No such file or directory: ~/data/hardcover/"**
- Create the data directory: `mkdir -p ~/data/hardcover`

**"ERROR: Set HARDCOVER_API_TOKEN in .env file!"**
- Create `.env` file in project root with: `HARDCOVER_API_TOKEN=your_token_here`

**GraphQL errors or rate limit exceeded**
- Wait 60 seconds and retry
- Reduce batch frequency (increase sleep time between batches)

**Script interrupted mid-batch**
- Safe to re-run - progress only updates after all 4 phases complete
- Partial data from interrupted batch will be overwritten on next run

## Hybrid Recommendation System (PRODUCTION - RECOMMENDED)

### Quick Start

**Generate recommendations (production system):**
```bash
python3 recommend.py                    # 3 random users
python3 recommend.py --user_id 62408    # Specific user
python3 recommend.py --top 20           # Top 20 recommendations
```

**Performance: 8.83% Precision@10**
- 4x better than pure collaborative filtering (2.16%)
- 17% better than popularity baseline (7.56%)
- Uses optimal 50/50 ensemble of popularity + collaborative filtering

### How It Works

The hybrid system combines three powerful signals:

1. **Popularity Component (50% weight)**
   - Recommends books many users have read
   - Simple but effective baseline
   - Ensures quality recommendations

2. **Collaborative Filtering (50% weight)**
   - Personalized based on similar users
   - Matrix factorization with 20 features, λ=1.0
   - Learns user preferences

3. **Implicit Feedback**
   - Want to read → 0.3 (weak positive signal)
   - Currently reading → 0.7 (likely positive)
   - Read with rating ≥3 → 1.0 (strong positive)
   - Read with rating <3 or DNF → 0.0 (negative)

**Key insight:** Using "want to read" and "currently reading" adds **11,000+ extra signals** (75% more data), dramatically improving collaborative filtering from 2.45% to 5.31% precision.

### Performance Comparison

| Approach | Precision@10 | Notes |
|----------|--------------|-------|
| Pure collaborative | 2.16% | Learns global bias, not preferences |
| Improved collaborative | 2.45% | With filtering + normalization |
| Popularity baseline | 7.56% | Simple but effective |
| **Hybrid (50/50)** | **8.83%** | **BEST - Production system** |

### When to Use What

**Use `recommend.py` (hybrid) for production:**
- Best overall performance (8.83%)
- Balances popularity with personalization
- Ready for deployment

**Use experimentation scripts for research:**
- `hardcover_collab_filter.py` - Test different collaborative filtering configs
- `improved_collab_filter.py` - Improved collaborative with normalization
- `evaluate_recommendations.py` - Comprehensive metrics (Precision, Recall, NDCG)
- `hybrid_recommender.py` - Test different hybrid weight combinations

### Why Hybrid Beats Pure Approaches

**Sparsity problem:** With 95.75% sparse data and 95% likes, collaborative filtering alone struggles to learn individual preferences.

**Solution:** Combine proven popularity (what most people like) with personalization (what YOU specifically might like).

**Result:**
- Popularity provides safe, quality baseline
- Collaborative adds personalization on top
- 50/50 balance prevents either from dominating
- Implicit feedback provides crucial extra training signal

### Data Filtering

Production system filters to:
- Users with ≥20 ratings (246/1000 users kept)
- Books with ≥5 users (2,547/45,203 books kept)
- Results in 26,598 training signals (95.75% sparse)

**Why filter aggressively?**
- Users with <20 ratings provide insufficient signal
- Books with <5 users likely niche/noise
- Better to have accurate recommendations for active users than poor recommendations for everyone

## Implicit Feedback and Sigmoid Function

### Why Different Book Statuses Have Different Weights

**The Problem:** Not all interactions are equal signals of preference.

Traditional collaborative filtering uses explicit ratings (1-5 stars). Hardcover provides different types of implicit feedback:
- Books you've **read** (with ratings)
- Books you're **currently reading**
- Books you **want to read**
- Books you **did not finish**

**Solution: Implicit Feedback Weighting**

We assign different weights based on signal strength:

```python
# Strong positive signal
Read with rating ≥3 → 1.0 (definitely liked it)

# Moderate positive signal
Currently reading → 0.7 (probably enjoying it, or wouldn't continue)

# Weak positive signal
Want to read → 0.3 (interested, but haven't committed yet)

# Negative signals
Read with rating <3 → 0.0 (didn't like it)
Did not finish (DNF) → 0.0 (strong dislike - couldn't finish)

# Unknown (masked out during training)
Unread → 0.5 (completely ignored via R mask matrix)
```

**Impact:** This scheme added **11,424 extra training signals** (75% more data), improving collaborative filtering precision from 2.45% to 5.31%.

### Why We Need the Sigmoid Function

**The sigmoid function σ(x) = 1/(1+e^(-x)) is essential for three reasons:**

**1. Maps to Probability Range [0,1]**

The raw model output X·W^T + b can be any real number (-∞ to +∞). Sigmoid squashes this to [0,1]:

```python
raw_logit = X @ W.T + b     # Can be -100 or +100 or anything
probability = σ(raw_logit)  # Always between 0 and 1
```

This makes the output interpretable as: "probability user will like this book"

**2. Handles Implicit Feedback Values**

Our training targets aren't just {0, 1} - they're also {0.3, 0.7}:

```python
# Without sigmoid (using raw logits):
loss = (logit - 0.7)²  # ❌ Doesn't make sense! Logit can be 100

# With sigmoid (using probabilities):
loss = (σ(logit) - 0.7)²  # ✅ Makes sense! Both values in [0,1]
```

The sigmoid ensures we're comparing probabilities to probabilities.

**3. Enables Binary Cross-Entropy Loss**

Binary cross-entropy loss requires probabilities as input:

```python
BCE(y_true, y_pred) = -[y_true·log(y_pred) + (1-y_true)·log(1-y_pred)]
```

This only works when `y_pred` is a probability (0 to 1). Without sigmoid, we'd get:
- `log(negative number)` → undefined
- `log(number > 1)` → wrong interpretation

**Visual Example:**

```
User hasn't read Harry Potter yet.
Model computes: logit = 2.5 (raw score)

Without sigmoid: "Recommendation score: 2.5" (what does this mean?)
With sigmoid: σ(2.5) = 0.92 = "92% probability user will like it" ✓

Compare to training data:
- Similar user currently reading it → target = 0.7
- Loss = (0.92 - 0.7)² → model slightly over-predicted
- Gradient descent adjusts parameters to bring 0.92 closer to 0.7
```

**Why Not Other Activation Functions?**

| Function | Range | Why Not Use It? |
|----------|-------|-----------------|
| **Sigmoid** | (0, 1) | ✅ Perfect for probabilities |
| Tanh | (-1, 1) | ❌ Negative values don't match {0, 0.3, 0.7, 1} targets |
| ReLU | [0, ∞) | ❌ Can output >1, not a probability |
| Linear | (-∞, ∞) | ❌ Can't interpret as probability |

**Summary:**
- Sigmoid converts raw scores → probabilities
- Probabilities match our implicit feedback weights (0.3, 0.7, 1.0)
- Binary cross-entropy requires probabilities
- Result: Smooth, interpretable, mathematically sound predictions

## Collaborative Filtering Recommendation System (Research/Experimentation)

### Overview

This section covers the pure collaborative filtering approach and experimentation. For production use, see the Hybrid Recommendation System above.

The system uses matrix factorization-based collaborative filtering with TensorFlow to predict whether users will like books they haven't read yet.

### Running Experiments

**Dependencies:**
```bash
pip install tensorflow numpy
```

**Grid search optimization:**
```bash
python3 hardcover_collab_filter.py
```

**Improved collaborative filtering:**
```bash
python3 improved_collab_filter.py
```

**Comprehensive evaluation:**
```bash
python3 evaluate_recommendations.py
```

### Optimal Configuration (from grid search)

After testing 16 different configurations, the optimal parameters are:
- **Number of features:** 5
- **Regularization (λ):** 5
- **Test accuracy:** 94.69%
- **Training time:** ~12 seconds (200 iterations on M4 Mac)

**Key finding:** Fewer features (5) performed as well as more (10, 15, 20), suggesting the data has relatively simple patterns.

### Dataset Characteristics vs Netflix

**Hardcover data (1000 users):**
- **Books (total):** 45,203
- **Books (≥2 users):** 11,168 (24.7%)
- **Matrix size:** 11,168 × 1,000 = 11.2M potential entries
- **Actual ratings:** ~25,132 (after filtering)
- **Matrix sparsity:** 99.77%
- **Likes/Dislikes ratio:** 95% likes, 5% dislikes
- **Average books per user:** 85.1

**Netflix data (from reference implementation):**
- Smaller subset for learning (~1,000-2,000 movies)
- Similar sparsity but more balanced ratings
- 5-star rating scale (1-5)

**Critical difference:** Hardcover is **significantly sparser** even after filtering to books with ≥2 users. This justifies:
- Conservative feature count (5-10)
- Higher regularization (λ=5-10)
- Binary classification instead of regression

### Algorithm Choice: {0,1} vs {-1,1}

**This implementation uses {0, 1} binary classification:**

```python
0 = dislike (read with rating <3, or DNF)
1 = like (read with rating ≥3)
0.5 = unrated (ignored in loss function)
```

**Alternative {-1, 1} approach:**

```python
-1 = dislike
1 = like
0 = unrated (ignored in loss)
```

**Comparison:**

| Aspect | {0,1} System (USED) | {-1,1} System |
|--------|---------------------|---------------|
| **Activation** | Sigmoid (0 to 1) | Tanh (-1 to 1) |
| **Loss function** | Binary cross-entropy | Hinge loss or MSE |
| **Interpretation** | Probability of liking | Signed preference |
| **Neutral value** | 0.5 | 0 |
| **Symmetry** | Not symmetric | Symmetric around 0 |
| **Stability** | Better for sparse data | Better for balanced data |

**Why {0,1} is better for Hardcover:**
- Predicts **probability** of liking (more interpretable)
- Sigmoid + binary cross-entropy more stable with sparse data
- Natural fit: "User has 85% probability of liking this book"
- Works well with highly imbalanced data (95% likes, 5% dislikes)

**When {-1,1} would be better:**
- Equal emphasis on likes and dislikes
- Using SVD-based methods instead of gradient descent
- More balanced positive/negative feedback

### Data Normalization

**Short answer: NO normalization needed for binary classification.**

**Detailed explanation:**

**5-star rating systems (like Netflix `my_netflix.py`):**
- ✅ **YES, normalize** by subtracting per-movie mean
- Removes bias where some movies rated higher overall
- Example: Pixar movies avg 4.5 stars, Horror avg 3.2 stars
- Normalization makes features learn preferences, not absolute popularity

**Binary systems (Hardcover):**
- ❌ **NO normalization needed**
- Binary cross-entropy loss handles scale naturally
- Bias term `b` captures user-specific tendencies
- No rating scale to normalize (just 0 or 1)

**Why Hardcover doesn't need normalization:**
1. Using sigmoid cross-entropy, not MSE
2. Bias term `b` learns per-user baseline propensity to like books
3. Not predicting numeric ratings (1-5), just binary like/dislike
4. High sparsity makes computing meaningful per-book means difficult

**When you WOULD normalize:**
If switching to 5-star predictions using actual ratings, then:
```python
Ynorm = Y - mean_rating_per_book
```
Like the Netflix 5-star system does in `normalizeRatings()`.

### Architecture Details

**Model structure:**
```python
predicted_logit = X @ W^T + b
predicted_probability = sigmoid(predicted_logit)
binary_prediction = 1 if probability >= 0.5 else 0
```

Where:
- **X:** Book feature matrix (11,168 books × 5 features)
- **W:** User feature matrix (1,000 users × 5 features)
- **b:** User bias vector (1 × 1,000 users)

**Loss function:**
```python
J = sum(sigmoid_cross_entropy(logits, Y) * R) + (λ/2) * (||X||² + ||W||²)
```

Where:
- **Y:** Binary rating matrix (0 or 1 for rated items, 0.5 for unrated)
- **R:** Indicator matrix (1 if rated, 0 if unrated)
- **λ:** Regularization parameter (5 is optimal)

**Training:**
- Optimizer: Adam with learning rate 0.1
- Iterations: 200
- Train/test split: 80/20
- Training time: ~12 seconds on M4 Mac

### Performance Metrics

**Test accuracy: 94.69%**

This is very high, but consider:
- Data is 95% likes, 5% dislikes
- Naive baseline (always predict "like") = 95% accuracy
- Model achieves 94.69% by learning actual patterns
- Real value is in **ranking** recommendations, not just binary prediction

**Interpretation:**
- Model successfully identifies most books users will like
- High accuracy reflects data skew, not model complexity
- Ranking by probability scores more useful than binary predictions

### Data Preparation for Collaborative Filtering

**Filter to books with ≥2 users:**
```python
# Reduces from 45,203 to 11,168 books
# Improves sparsity from 99.8% to 99.77%
MIN_USERS_PER_BOOK = 2
```

**Binary rating transformation:**
```python
# Read with rating ≥3 → 1 (like)
# Read with rating <3 → 0 (dislike)
# Did not finish (status_id=5) → 0 (dislike)
# Want to read, Currently reading → 0.5 (unrated, ignored in loss)
```

**Train/test split:**
- 80% of ratings for training
- 20% held out for testing
- Random seed=42 for reproducibility

### Expected Performance at Scale

Current (1000 users):
- 11,168 books with ≥2 users
- 94.69% accuracy
- ~12 second training time

At 10,000 users (projected):
- ~50,000-80,000 books with ≥2 users
- Expected accuracy: 92-95%
- Training time: ~2-3 minutes
- May need to increase features to 10-15

### For Recommendation System Development

Current dataset (1000 users):
- 45,203 unique books
- 11,168 books (24.7%) have 2+ users - **use these for collaborative filtering**
- Average 85.1 books per user
- Matrix sparsity: ~99.8% (filter to books with ≥2 users for denser matrix)

Recommended: Filter to books with ≥2 or ≥3 users before building recommendation model.

## User Clustering for Friend Finding

### Quick Start

**Find optimal clusters:**
```bash
python3 cluster_users.py
```

**Find book friends for a user:**
```bash
python3 find_book_friends.py
python3 find_book_friends.py --user_id 62408
python3 find_book_friends.py --num_matches 10
```

**Run web app (friend finding interface):**
```bash
python3 app.py
# Open http://localhost:8000 in browser
```

### How Clustering Works

**Goal:** Group users with similar reading preferences to help them find potential friends who share their taste in books.

**Algorithm:** K-means clustering on L2-normalized user feature vectors

**Process:**
1. Extract user feature vectors W from collaborative filtering model (246 users × 20 features)
2. Normalize vectors to unit length (L2 normalization)
3. Test K=3 to K=15 clusters using silhouette score as metric
4. Select optimal K based on highest silhouette score

**Result:** K=15 clusters with silhouette score of 0.279

### Optimal Cluster Count

**Current configuration:** K=15 clusters (configured in `precompute_all.py`)

**Silhouette score interpretation:**
- Measures how well-separated clusters are
- Range: -1 (poor) to 1 (perfect)
- 0.279 indicates good separation for high-dimensional, sparse data

### Finding Book Friends

**Similarity metric:** Cosine similarity (dot product of normalized user vectors)

```python
similarity(user_i, user_j) = w_i · w_j / (||w_i|| ||w_j||)
```

**Process:**
1. Find user's cluster assignment
2. Calculate cosine similarity with all users in same cluster
3. Rank by similarity score (higher = more similar tastes)
4. Find shared books as conversation starters

**Output:**
- Top N most compatible users in same cluster
- Compatibility score (0-100%)
- Shared books both users have read
- User reading statistics

### Web Application (Flask)

**Location:** `webapp/` directory (deployment-ready version)

**Running locally:**
```bash
cd webapp
python3 app.py
# Access at http://localhost:8001
```

**Port:** 8001 (configurable via PORT environment variable)

**Features:**
- Searchable user selection (type to filter by name or username)
- Friend matching within user's cluster
- Compatibility scoring (cosine similarity)
- Shared books display
- Cluster view showing all members
- **Chat system** - Message top 10 book friends

**Routes:**
- `GET /` - Home page with user selection
- `POST /find_friends` - Find compatible users
- `GET /cluster/<cluster_id>` - View all cluster members
- `GET /chat?user_id=N` - Chat inbox
- `GET/POST /chat/<other_user_id>?user_id=N` - Chat conversation
- `GET /chat/<other_user_id>/messages?user_id=N&since=T` - Poll new messages
- `GET /chat/unread_count?user_id=N` - Get unread count

**Chat System (`chat_db.py`):**
- SQLite database stored in `webapp/data/chat.db`
- Access control: Users can only chat with their top 10 matches
- Features: Send/receive messages, unread counts, conversation history

**Templates:**
- `templates/index.html` - User selection page
- `templates/results.html` - Friend recommendations with compatibility bars
- `templates/cluster.html` - Full cluster member list
- `templates/chat_inbox.html` - List of all conversations
- `templates/chat_conversation.html` - Individual chat thread

### Data Files

**Pre-computed files in `webapp/data/`:**
- `recommendations.json` - All friend matches and compatibility scores
- `users.json` - User list for dropdown
- `clusters.json` - Cluster membership info

**Generated by `precompute_all.py`** before deployment.

## Deployment

### Production URL
**https://book-friend-finder-770103525576.us-central1.run.app**

### Deploying Updates

The webapp is deployed to Google Cloud Run. To deploy updates:

**1. Download more user data (optional):**
```bash
# Download 500 more users (20 batches of 25)
for i in {1..20}; do
  echo "=== Batch $i/20 ==="
  python3 scripts/process_batch.py
  sleep 15
done
```

**2. Retrain the model and precompute recommendations:**
```bash
# Find optimal clusters (optional - currently K=15)
python3 cluster_users.py

# Precompute all recommendations (required before deploy)
python3 precompute_all.py
```

This generates `webapp/data/recommendations.json`, `users.json`, and `clusters.json`.

**3. Deploy to Cloud Run:**
```bash
cd webapp
gcloud run deploy book-friend-finder --source . --region us-central1 --allow-unauthenticated
```

### Deployment Requirements

- Google Cloud SDK (`gcloud`) installed and authenticated
- Project set to `book-friend-finder`: `gcloud config set project book-friend-finder`
- The `webapp/` directory contains:
  - `app.py` - Flask application
  - `requirements.txt` - Python dependencies (Flask, gunicorn)
  - `Procfile` - Gunicorn startup command
  - `data/` - Pre-computed JSON files
  - `templates/` - HTML templates

### Notes

- Chat messages are stored in SQLite (`webapp/data/chat.db`) which persists between deploys on the same instance but may be lost if Cloud Run provisions a new instance
- Pre-computed recommendations mean no ML dependencies needed at runtime
- The webapp auto-detects `PORT` environment variable (Cloud Run sets this)

## Utility Scripts

### Generate Statistics

**File:** `generate_stats.py`

**Purpose:** Comprehensive statistics about the recommendation system and scalability analysis

**Output:** `~/data/hardcover/system_stats.txt`

**Metrics included:**
- Dataset statistics (users, books, sparsity)
- Training performance (time, iterations)
- Scalability projections (10K, 100K users)
- Cluster distribution
- Precision/Recall metrics

**Usage:**
```bash
python3 generate_stats.py
cat ~/data/hardcover/system_stats.txt
```

### Visualize Results

**File:** `visualize_results.py`

**Purpose:** Text-based comparison of all recommendation approaches

**Shows progression:**
- Pure collaborative filtering: 2.16%
- Improved collaborative: 2.45%
- Popularity baseline: 7.56%
- Hybrid (50/50): 8.83%

**Usage:**
```bash
python3 visualize_results.py
```

### Create User List

**File:** `create_user_list.py`

**Purpose:** Generate list of active users (≥20 ratings) for web interface

**Output:** `~/data/hardcover/active_users.txt`

**Format:** One username per line (246 users)

**Usage:**
```bash
python3 create_user_list.py
```

## Key Files Overview

### Data Collection
- `process_batch.py` - Main orchestrator (4-phase atomic batch processing)
- `progress.json` - Batch progress tracking

### Recommendation System
- `recommend.py` - **PRODUCTION** hybrid recommender (8.83% precision)
- `precompute_all.py` - Generate pre-computed data for webapp
- `hardcover_collab_filter.py` - Grid search for optimal config
- `improved_collab_filter.py` - Improved collaborative with filtering
- `hybrid_recommender.py` - Experimental hybrid weight testing
- `evaluate_recommendations.py` - Comprehensive metrics (Precision, Recall, NDCG)

### Friend Finding
- `cluster_users.py` - K-means clustering analysis
- `find_book_friends.py` - CLI friend matching tool

### Web Application (`webapp/`)
- `app.py` - Flask web application (port 8001)
- `chat_db.py` - SQLite chat database module
- `templates/` - HTML templates (index, results, cluster, chat_inbox, chat_conversation)
- `data/` - Pre-computed JSON files and chat.db
- `requirements.txt` - Flask + gunicorn dependencies
- `Dockerfile` - Container configuration for Cloud Run

### Utilities
- `generate_stats.py` - Generate comprehensive statistics
- `visualize_results.py` - Text-based results comparison
- `create_user_list.py` - Generate active users list

### Data Files
Located in `~/data/hardcover/`:
- `users.json` - User profiles
- `user_books.json` - User reading lists (~100MB+)
- `books_users.json` - Inverted book→users mapping (~60MB+)
- `active_users.txt` - Active user list

## Current Dataset Stats

- **3,000 total users** downloaded (120 batches)
- **859 active users** (with ≥20 ratings)
- **111,407 books** total
- **55.4% books** have >1 user (good for collaborative filtering)
- **15 clusters** for friend matching
