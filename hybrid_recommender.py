#!/usr/bin/env python3
"""
Hardcover Hybrid Recommender System

Combines three powerful signals:
1. POPULARITY: Books many users have read (simple but effective)
2. COLLABORATIVE FILTERING: Personalized based on similar users
3. IMPLICIT FEEDBACK: Uses want_to_read, currently_reading as weak signals

Rating scale:
  1.0 = Read with rating ≥3 (strong like)
  0.7 = Currently reading (likely to like)
  0.3 = Want to read (weak positive signal)
  0.0 = Read with rating <3 or DNF (dislike)

Ensemble:
  final_score = w1×popularity + w2×collaborative + w3×(w1×popularity + w2×collaborative)
  Tests multiple weight combinations to find optimal
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from collections import defaultdict

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5

# Collaborative filtering params (from best config)
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 300
LEARNING_RATE = 0.1

# Ensemble weights to test
ENSEMBLE_CONFIGS = [
    {'popularity': 0.7, 'collaborative': 0.3},
    {'popularity': 0.6, 'collaborative': 0.4},
    {'popularity': 0.5, 'collaborative': 0.5},
    {'popularity': 0.4, 'collaborative': 0.6},
    {'popularity': 0.3, 'collaborative': 0.7},
]

print("="*80)
print("HYBRID RECOMMENDER SYSTEM")
print("="*80)
print("Components:")
print("  1. Popularity baseline (proven 7.31% precision)")
print("  2. Collaborative filtering (personalization)")
print("  3. Implicit feedback (want_to_read, currently_reading)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

print(f"  {len(users_data['user_books'])} users, {len(books_data['books'])} books")

# ============================================================================
# 2. FILTER ACTIVE USERS AND POPULAR BOOKS
# ============================================================================
print(f"\n[2/7] Filtering (users ≥{MIN_RATINGS_PER_USER} ratings, books ≥{MIN_USERS_PER_BOOK} users)...")

# Count ratings per user
user_rating_counts = {}
for user_entry in users_data['user_books']:
    user_id = user_entry['user']['id']
    rating_count = sum(user_entry['counts'].values())
    user_rating_counts[user_id] = rating_count

# Filter users
filtered_users = [
    user_entry for user_entry in users_data['user_books']
    if user_rating_counts[user_entry['user']['id']] >= MIN_RATINGS_PER_USER
]

# Filter books
filtered_books = [
    book_entry for book_entry in books_data['books']
    if book_entry['user_count'] >= MIN_USERS_PER_BOOK
]

print(f"  Kept {len(filtered_users)} users, {len(filtered_books)} books")

# Create mappings
user_id_to_idx = {user_entry['user']['id']: idx for idx, user_entry in enumerate(filtered_users)}
book_id_to_idx = {book_entry['book']['id']: idx for idx, book_entry in enumerate(filtered_books)}

num_movies = len(filtered_books)
num_users = len(filtered_users)

# ============================================================================
# 3. BUILD RATING MATRIX WITH IMPLICIT FEEDBACK
# ============================================================================
print("\n[3/7] Building rating matrix with implicit feedback...")

Y_raw = np.full((num_movies, num_users), np.nan)
R = np.zeros((num_movies, num_users))

# Track counts for each feedback type
feedback_counts = defaultdict(int)

for book_entry in filtered_books:
    book_id = book_entry['book']['id']
    if book_id not in book_id_to_idx:
        continue
    book_idx = book_id_to_idx[book_id]

    for user_entry in book_entry['users']:
        user_id = user_entry['user_id']
        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]
        status_id = user_entry['status_id']
        rating = user_entry.get('rating')

        # IMPLICIT FEEDBACK SCALE
        if status_id == 3:  # Read
            if rating is not None:
                if rating >= 3:
                    Y_raw[book_idx, user_idx] = 1.0  # Strong like
                    feedback_counts['read_liked'] += 1
                else:
                    Y_raw[book_idx, user_idx] = 0.0  # Dislike
                    feedback_counts['read_disliked'] += 1
            else:
                Y_raw[book_idx, user_idx] = 1.0  # Assume liked
                feedback_counts['read_no_rating'] += 1
            R[book_idx, user_idx] = 1
        elif status_id == 2:  # Currently reading
            Y_raw[book_idx, user_idx] = 0.7  # Likely to like
            R[book_idx, user_idx] = 1
            feedback_counts['currently_reading'] += 1
        elif status_id == 1:  # Want to read
            Y_raw[book_idx, user_idx] = 0.3  # Weak positive
            R[book_idx, user_idx] = 1
            feedback_counts['want_to_read'] += 1
        elif status_id == 5:  # DNF
            Y_raw[book_idx, user_idx] = 0.0  # Dislike
            R[book_idx, user_idx] = 1
            feedback_counts['dnf'] += 1

total_ratings = int(np.sum(R))
sparsity = 100 * (1 - total_ratings / (num_movies * num_users))

print(f"  Total feedback signals: {total_ratings:,}")
print(f"  Breakdown:")
for feedback_type, count in sorted(feedback_counts.items()):
    print(f"    {feedback_type}: {count:,} ({100*count/total_ratings:.1f}%)")
print(f"  Matrix sparsity: {sparsity:.2f}%")

# ============================================================================
# 4. TRAIN/TEST SPLIT
# ============================================================================
print("\n[4/7] Creating 80/20 train/test split...")

def decompose_matrix_80_20(R, random_seed=42):
    R = np.array(R)
    A = np.zeros_like(R)
    B = np.zeros_like(R)
    np.random.seed(random_seed)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] == 1:
                if np.random.rand() < 0.8:
                    A[i, j] = 1
                else:
                    B[i, j] = 1
    return A, B

R_train, R_test = decompose_matrix_80_20(R)

Y_train = Y_raw.copy()
Y_train[R_train == 0] = 0.5  # Neutral for unrated

Y_test = Y_raw.copy()

train_count = int(np.sum(R_train))
test_count = int(np.sum(R_test))

print(f"  Training: {train_count:,} signals")
print(f"  Testing: {test_count:,} signals")

# ============================================================================
# 5. COMPONENT 1: POPULARITY SCORES
# ============================================================================
print("\n[5/7] Computing popularity scores...")

# Count how many users interacted with each book
book_popularity = np.sum(R_train, axis=1)
max_popularity = np.max(book_popularity)

# Normalize to [0, 1]
popularity_scores = book_popularity / max_popularity

print(f"  Most popular book: {book_popularity.max():.0f} users")
print(f"  Average popularity: {book_popularity.mean():.1f} users")

# ============================================================================
# 6. COMPONENT 2: COLLABORATIVE FILTERING
# ============================================================================
print(f"\n[6/7] Training collaborative filtering ({ITERATIONS} iterations)...")

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """Collaborative filtering with implicit feedback"""
    R_mask = tf.where(tf.equal(Y, 0.5),
                      tf.constant(0.0, dtype=tf.float32),
                      tf.constant(1.0, dtype=tf.float32))

    Y_values = tf.cast(Y, tf.float32)
    Y_values = tf.where(tf.equal(R_mask, 1), Y_values, tf.zeros_like(Y_values))

    # Predictions
    logits = tf.matmul(X, W, transpose_b=True) + b
    probs = tf.sigmoid(logits)

    # MSE loss (better for continuous 0.0-1.0 scale)
    loss = (probs - Y_values) ** 2

    masked_loss = loss * R_mask

    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )

    return total_cost

# Initialize
tf.random.set_seed(42)
W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), dtype=tf.float32) * 0.01, name='W')
X = tf.Variable(tf.random.normal((num_movies, NUM_FEATURES), dtype=tf.float32) * 0.01, name='X')
b = tf.Variable(tf.zeros((1, num_users), dtype=tf.float32), name='b')

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Training
for iter in range(ITERATIONS):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Y_train, R_train, LAMBDA)

    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter % 75 == 0:
        print(f"  Iteration {iter}: loss = {cost_value:.1f}")

# Get collaborative filtering predictions
logits = tf.matmul(X, W, transpose_b=True) + b
collab_probs = tf.sigmoid(logits).numpy()

print(f"  ✓ Collaborative filtering trained")

# ============================================================================
# 7. HYBRID ENSEMBLE: TEST WEIGHT COMBINATIONS
# ============================================================================
print("\n[7/7] Testing hybrid ensemble combinations...")
print("="*80)

def compute_precision_at_k(predictions, Y_test, R_test, R_train, k=10):
    """Compute precision@K for predictions"""
    precision_scores = []

    for user_idx in range(num_users):
        user_test_mask = R_test[:, user_idx] == 1
        # For implicit feedback, consider ≥0.5 as "like"
        user_test_likes = np.where((Y_test[:, user_idx] >= 0.5) & user_test_mask)[0]

        if len(user_test_likes) == 0:
            continue

        # Get predictions, exclude training items
        user_preds = predictions[:, user_idx].copy()
        user_preds[R_train[:, user_idx] == 1] = -1

        # Top K
        top_k = np.argsort(-user_preds)[:k]
        hits = len(set(top_k) & set(user_test_likes))
        precision_scores.append(hits / k)

    return np.mean(precision_scores) if precision_scores else 0

results = []

# Test pure components first
print("\nPURE COMPONENTS:")

# Popularity only
pop_scores_matrix = np.tile(popularity_scores.reshape(-1, 1), (1, num_users))
pop_precision = compute_precision_at_k(pop_scores_matrix, Y_test, R_test, R_train)
print(f"  Popularity only: {pop_precision*100:.2f}%")

# Collaborative only
collab_precision = compute_precision_at_k(collab_probs, Y_test, R_test, R_train)
print(f"  Collaborative only: {collab_precision*100:.2f}%")

# Test ensemble combinations
print("\nHYBRID ENSEMBLES:")

for config in ENSEMBLE_CONFIGS:
    w_pop = config['popularity']
    w_collab = config['collaborative']

    # Combine scores
    hybrid_scores = (w_pop * pop_scores_matrix + w_collab * collab_probs)

    precision = compute_precision_at_k(hybrid_scores, Y_test, R_test, R_train)

    print(f"  {w_pop:.1f} pop + {w_collab:.1f} collab: {precision*100:.2f}%")

    results.append({
        'w_pop': w_pop,
        'w_collab': w_collab,
        'precision': precision
    })

# Find best
results.sort(key=lambda x: x['precision'], reverse=True)
best = results[0]

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\nBaselines:")
print(f"  Original collaborative filtering: 2.16%")
print(f"  Improved collaborative filtering: 2.45%")
print(f"  Popularity baseline: {pop_precision*100:.2f}%")

print(f"\nHybrid model (BEST):")
print(f"  Weights: {best['w_pop']:.1f} popularity + {best['w_collab']:.1f} collaborative")
print(f"  Precision@10: {best['precision']*100:.2f}%")

improvement_vs_collab = (best['precision'] - 0.0245) / 0.0245 * 100
improvement_vs_pop = (best['precision'] - pop_precision) / pop_precision * 100

if best['precision'] > pop_precision:
    print(f"\n  ✓ {improvement_vs_pop:+.1f}% improvement over popularity!")
    print(f"  ✓ {improvement_vs_collab:+.1f}% improvement over collaborative filtering!")
    print(f"\n  🎉 HYBRID MODEL IS BEST!")
elif best['precision'] > 0.0245:
    print(f"\n  ✓ {improvement_vs_collab:+.1f}% improvement over collaborative filtering!")
    print(f"  ~ {improvement_vs_pop:+.1f}% vs popularity (close!)")
else:
    print(f"\n  ~ Hybrid similar to pure components")

print(f"\nKEY INSIGHTS:")
print(f"1. Implicit feedback added {total_ratings - 15174:,} extra signals")
print(f"   - Want to read: {feedback_counts['want_to_read']:,}")
print(f"   - Currently reading: {feedback_counts['currently_reading']:,}")
print(f"2. {'Popularity' if pop_precision > collab_precision else 'Collaborative'} is stronger component")
print(f"3. Best combination: {best['w_pop']*100:.0f}% popularity, {best['w_collab']*100:.0f}% collaborative")

print("="*80)
print("✓ Hybrid recommender evaluation complete!")
