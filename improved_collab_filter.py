#!/usr/bin/env python3
"""
Improved Hardcover Collaborative Filtering

Improvements over baseline:
1. Filter to users with ≥20 ratings (more signal per user)
2. Filter to books with ≥5 users (reduce noise)
3. Per-user normalization (learn preferences, not tendencies)
4. Class weighting (handle 95% like / 5% dislike imbalance)
5. Test multiple configurations (features, regularization)
6. More training iterations (300)
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

# IMPROVED FILTERS
MIN_RATINGS_PER_USER = 20  # Up from no filter
MIN_USERS_PER_BOOK = 5     # Up from 2

# EXPERIMENT CONFIGURATIONS
CONFIGS = [
    {'features': 10, 'lambda': 0.5, 'iterations': 300},
    {'features': 10, 'lambda': 1.0, 'iterations': 300},
    {'features': 15, 'lambda': 0.5, 'iterations': 300},
    {'features': 15, 'lambda': 1.0, 'iterations': 300},
    {'features': 20, 'lambda': 0.5, 'iterations': 300},
    {'features': 20, 'lambda': 1.0, 'iterations': 300},
]

LEARNING_RATE = 0.1
CLASS_WEIGHT_DISLIKE = 10.0  # Weight dislikes 10x more (they're only 5% of data)

print("="*80)
print("IMPROVED COLLABORATIVE FILTERING")
print("="*80)
print(f"Improvements:")
print(f"  - Filter users with ≥{MIN_RATINGS_PER_USER} ratings")
print(f"  - Filter books with ≥{MIN_USERS_PER_BOOK} users")
print(f"  - Per-user normalization")
print(f"  - Class weighting (dislikes weighted {CLASS_WEIGHT_DISLIKE}x)")
print(f"  - Testing {len(CONFIGS)} configurations")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

print(f"  Loaded {len(users_data['user_books'])} users, {len(books_data['books'])} books")

# ============================================================================
# 2. FILTER USERS WITH ≥ MIN_RATINGS_PER_USER
# ============================================================================
print(f"\n[2/6] Filtering to active users (≥{MIN_RATINGS_PER_USER} ratings)...")

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

print(f"  Kept {len(filtered_users)}/{len(users_data['user_books'])} users")
print(f"  Average ratings per filtered user: {np.mean([user_rating_counts[u['user']['id']] for u in filtered_users]):.1f}")

# ============================================================================
# 3. FILTER BOOKS WITH ≥ MIN_USERS_PER_BOOK
# ============================================================================
print(f"\n[3/6] Filtering to popular books (≥{MIN_USERS_PER_BOOK} users)...")

filtered_books = [
    book_entry for book_entry in books_data['books']
    if book_entry['user_count'] >= MIN_USERS_PER_BOOK
]

print(f"  Kept {len(filtered_books)}/{len(books_data['books'])} books")

# Create mappings
user_id_to_idx = {user_entry['user']['id']: idx for idx, user_entry in enumerate(filtered_users)}
book_id_to_idx = {book_entry['book']['id']: idx for idx, book_entry in enumerate(filtered_books)}

num_movies = len(filtered_books)
num_users = len(filtered_users)

print(f"  Final matrix: {num_movies} books × {num_users} users = {num_movies * num_users:,} entries")

# ============================================================================
# 4. BUILD RATING MATRICES WITH PER-USER NORMALIZATION
# ============================================================================
print("\n[4/6] Building rating matrices with normalization...")

Y_raw = np.full((num_movies, num_users), np.nan)  # Use NaN for unrated
R = np.zeros((num_movies, num_users))

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

        # Binary rating
        if status_id == 3:  # Read
            if rating is not None:
                Y_raw[book_idx, user_idx] = 1 if rating >= 3 else 0
                R[book_idx, user_idx] = 1
            else:
                Y_raw[book_idx, user_idx] = 1
                R[book_idx, user_idx] = 1
        elif status_id == 5:  # DNF
            Y_raw[book_idx, user_idx] = 0
            R[book_idx, user_idx] = 1

# Per-user normalization: subtract each user's mean
Y_normalized = Y_raw.copy()
user_means = np.zeros(num_users)

for user_idx in range(num_users):
    user_mask = R[:, user_idx] == 1
    if np.sum(user_mask) > 0:
        user_ratings = Y_raw[user_mask, user_idx]
        user_mean = np.mean(user_ratings)
        user_means[user_idx] = user_mean
        Y_normalized[user_mask, user_idx] -= user_mean

# Replace NaN with 0 (neutral for normalized data)
Y = np.nan_to_num(Y_normalized, nan=0.0)

total_ratings = int(np.sum(R))
num_likes = int(np.sum(Y_raw[R == 1] == 1))
num_dislikes = int(np.sum(Y_raw[R == 1] == 0))
sparsity = 100 * (1 - total_ratings / (num_movies * num_users))

print(f"  Total ratings: {total_ratings:,}")
print(f"    Likes: {num_likes:,} ({100*num_likes/total_ratings:.1f}%)")
print(f"    Dislikes: {num_dislikes:,} ({100*num_dislikes/total_ratings:.1f}%)")
print(f"  Matrix sparsity: {sparsity:.2f}%")
print(f"  User mean rating range: [{np.min(user_means):.2f}, {np.max(user_means):.2f}]")

# ============================================================================
# 5. CREATE TRAIN/TEST SPLIT
# ============================================================================
print("\n[5/6] Creating 80/20 train/test split...")

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

# For normalized data, keep zeros for unrated (not 0.5)
Y_train = Y.copy()
Y_train[R_train == 0] = 0

Y_test = Y.copy()
Y_test[R_test == 0] = 0

train_count = int(np.sum(R_train))
test_count = int(np.sum(R_test))

print(f"  Training: {train_count:,} ratings")
print(f"  Testing: {test_count:,} ratings")

# ============================================================================
# 6. DEFINE IMPROVED COST FUNCTION WITH CLASS WEIGHTING
# ============================================================================

def cofi_cost_func_weighted(X, W, b, Y, Y_raw, R, user_means, lambda_, class_weight_dislike=1.0):
    """
    Improved cost function with:
    - Per-user normalization
    - Class weighting for imbalanced data
    """
    # Mask for rated items
    R_mask = tf.cast(R, tf.float32)

    # Get original binary labels (0 or 1)
    Y_binary = tf.cast(Y_raw, tf.float32)
    Y_binary = tf.where(tf.equal(R_mask, 1), Y_binary, tf.zeros_like(Y_binary))

    # Compute predictions (normalized space)
    logits_normalized = tf.matmul(X, W, transpose_b=True) + b

    # Convert to original space by adding user means
    user_means_expanded = tf.expand_dims(tf.cast(user_means, tf.float32), 0)
    logits = logits_normalized + user_means_expanded

    # Binary cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_binary, logits=logits)

    # Apply class weighting
    weights = tf.where(tf.equal(Y_binary, 0),
                       tf.constant(class_weight_dislike, dtype=tf.float32),
                       tf.constant(1.0, dtype=tf.float32))
    weighted_loss = loss * weights

    # Apply mask to ignore unrated items
    masked_loss = weighted_loss * R_mask

    # Total cost with regularization
    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )

    return total_cost

# ============================================================================
# 7. EVALUATION METRICS
# ============================================================================

def compute_metrics(X, W, b, Y_test, Y_raw, R_test, R_train, user_means):
    """Compute comprehensive metrics"""
    # Predictions in normalized space
    logits_normalized = tf.matmul(X, W, transpose_b=True) + b

    # Convert to original space
    user_means_expanded = np.expand_dims(user_means, 0)
    logits = logits_normalized.numpy() + user_means_expanded

    probabilities = 1 / (1 + np.exp(-logits))
    binary_predictions = (probabilities >= 0.5).astype(int)

    # Binary accuracy
    test_mask = (R_test == 1)
    matches = np.sum(binary_predictions[test_mask] == Y_raw[test_mask])
    accuracy = matches / np.sum(test_mask)

    # Precision@10
    precision_scores = []
    for user_idx in range(num_users):
        user_test_mask = R_test[:, user_idx] == 1
        user_test_likes = np.where((Y_raw[:, user_idx] == 1) & user_test_mask)[0]

        if len(user_test_likes) == 0:
            continue

        # Get probabilities, exclude training books
        user_probs = probabilities[:, user_idx].copy()
        user_probs[R_train[:, user_idx] == 1] = -1

        # Top 10 recommendations
        top_10 = np.argsort(-user_probs)[:10]
        hits = len(set(top_10) & set(user_test_likes))
        precision_scores.append(hits / 10)

    precision_at_10 = np.mean(precision_scores) if precision_scores else 0

    return accuracy, precision_at_10

# ============================================================================
# 8. TRAIN AND EVALUATE MULTIPLE CONFIGURATIONS
# ============================================================================
print("\n[6/6] Training and evaluating configurations...")
print("="*80)

results = []

# Convert to tensors once
Y_raw_for_training = Y_raw.copy()
Y_raw_for_training[R_train == 0] = 0  # Zero out test set

for config in CONFIGS:
    num_features = config['features']
    lambda_ = config['lambda']
    iterations = config['iterations']

    print(f"\nTesting: features={num_features}, λ={lambda_}, iterations={iterations}")

    # Initialize
    tf.random.set_seed(42)
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float32) * 0.01, name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float32) * 0.01, name='X')
    b = tf.Variable(tf.zeros((1, num_users), dtype=tf.float32), name='b')

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Training
    for iter in range(iterations):
        with tf.GradientTape() as tape:
            cost_value = cofi_cost_func_weighted(
                X, W, b, Y_train, Y_raw_for_training, R_train,
                user_means, lambda_, CLASS_WEIGHT_DISLIKE
            )

        grads = tape.gradient(cost_value, [X, W, b])
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        if iter % 75 == 0:
            print(f"  Iteration {iter}: loss = {cost_value:.1f}")

    # Evaluate
    accuracy, precision_at_10 = compute_metrics(
        X, W, b, Y_test, Y_raw, R_test, R_train, user_means
    )

    print(f"  ✓ Accuracy: {accuracy*100:.2f}%")
    print(f"  ✓ Precision@10: {precision_at_10*100:.2f}%")

    results.append({
        'features': num_features,
        'lambda': lambda_,
        'iterations': iterations,
        'accuracy': accuracy,
        'precision_at_10': precision_at_10
    })

# ============================================================================
# 9. RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Sort by precision@10
results.sort(key=lambda x: x['precision_at_10'], reverse=True)

print("\nTop Configurations (by Precision@10):\n")
print(f"{'Rank':<6} {'Features':<10} {'Lambda':<10} {'Precision@10':<15} {'Accuracy':<12}")
print("-"*80)

for i, result in enumerate(results, 1):
    print(f"{i:<6} {result['features']:<10} {result['lambda']:<10.1f} "
          f"{result['precision_at_10']*100:<14.2f}% {result['accuracy']*100:<11.2f}%")

best = results[0]
print("\n" + "="*80)
print("BEST CONFIGURATION:")
print(f"  Features: {best['features']}")
print(f"  Lambda: {best['lambda']}")
print(f"  Precision@10: {best['precision_at_10']*100:.2f}%")
print(f"  Accuracy: {best['accuracy']*100:.2f}%")
print("="*80)

print("\nCOMPARISON TO BASELINE:")
print(f"  Original model Precision@10: 2.16%")
print(f"  Improved model Precision@10: {best['precision_at_10']*100:.2f}%")
print(f"  Popularity baseline: 7.31%")

if best['precision_at_10'] > 0.0731:
    improvement = (best['precision_at_10'] - 0.0731) / 0.0731 * 100
    print(f"\n  ✓ {improvement:+.1f}% improvement over popularity baseline!")
elif best['precision_at_10'] > 0.0216:
    improvement = (best['precision_at_10'] - 0.0216) / 0.0216 * 100
    print(f"\n  ✓ {improvement:+.1f}% improvement over original model!")
else:
    print(f"\n  ⚠ Still worse than baselines - may need more data or different approach")

print("\n✓ Evaluation complete!")
