#!/usr/bin/env python3
"""
Hardcover Collaborative Filtering - Feature Count Optimization

Tests different numbers of latent features (5, 10, 15, 20) with various
regularization parameters to find optimal configuration for Hardcover data.

Uses binary rating system:
- Read (status_id=3) with rating ≥3 → 1 (like)
- Read with rating <3 → 0 (dislike)
- Did not finish (status_id=5) → 0 (dislike)
- Want to read, Currently reading → 0.5 (unrated)
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

# Filter settings
MIN_USERS_PER_BOOK = 2  # Only use books with at least N users

# Experiment grid
NUM_FEATURES_TO_TEST = [5, 10, 15, 20]
LAMBDA_VALUES = [1, 5, 10, 20]
ITERATIONS = 200
LEARNING_RATE = 0.1

print("="*80)
print("HARDCOVER COLLABORATIVE FILTERING - FEATURE OPTIMIZATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

total_books = books_data['metadata']['total_books']
total_users = users_data['metadata']['total_users']

print(f"  Loaded {total_users} users and {total_books} books")

# ============================================================================
# 2. FILTER BOOKS WITH ≥ MIN_USERS_PER_BOOK
# ============================================================================
print(f"\n[2/6] Filtering to books with ≥{MIN_USERS_PER_BOOK} users...")

filtered_books = [
    book_entry for book_entry in books_data['books']
    if book_entry['user_count'] >= MIN_USERS_PER_BOOK
]

num_filtered_books = len(filtered_books)
print(f"  Kept {num_filtered_books}/{total_books} books ({100*num_filtered_books/total_books:.1f}%)")

# Create book_id to index mapping
book_id_to_idx = {book_entry['book']['id']: idx for idx, book_entry in enumerate(filtered_books)}
book_idx_to_title = {idx: book_entry['book']['title'] for idx, book_entry in enumerate(filtered_books)}

# Create user_id to index mapping
user_id_to_idx = {user_entry['user']['id']: idx for idx, user_entry in enumerate(users_data['user_books'])}

num_movies = num_filtered_books  # Using 'movies' variable name to match Netflix code
num_users = total_users

print(f"  Matrix dimensions: {num_movies} books × {num_users} users")

# ============================================================================
# 3. BUILD RATING MATRIX Y AND INDICATOR MATRIX R
# ============================================================================
print("\n[3/6] Building rating matrices...")

Y = np.full((num_movies, num_users), 0.5)  # Start with all unrated (0.5)
R = np.zeros((num_movies, num_users))  # Indicator matrix

# Process each book
for book_entry in filtered_books:
    book_idx = book_id_to_idx[book_entry['book']['id']]

    for user_entry in book_entry['users']:
        user_id = user_entry['user_id']
        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]
        status_id = user_entry['status_id']
        rating = user_entry.get('rating')

        # Binary rating logic
        if status_id == 3:  # Read
            if rating is not None:
                if rating >= 3:
                    Y[book_idx, user_idx] = 1  # Like
                    R[book_idx, user_idx] = 1
                else:
                    Y[book_idx, user_idx] = 0  # Dislike
                    R[book_idx, user_idx] = 1
            else:
                # No rating, assume liked if they marked as read
                Y[book_idx, user_idx] = 1
                R[book_idx, user_idx] = 1
        elif status_id == 5:  # Did not finish
            Y[book_idx, user_idx] = 0  # Dislike
            R[book_idx, user_idx] = 1
        # Status 1 (want to read) and 2 (currently reading) remain 0.5 (unrated)

# Count ratings
total_ratings = int(np.sum(R))
num_likes = int(np.sum((Y == 1) & (R == 1)))
num_dislikes = int(np.sum((Y == 0) & (R == 1)))
sparsity = 100 * (1 - total_ratings / (num_movies * num_users))

print(f"  Total ratings: {total_ratings:,}")
print(f"    Likes (1): {num_likes:,} ({100*num_likes/total_ratings:.1f}%)")
print(f"    Dislikes (0): {num_dislikes:,} ({100*num_dislikes/total_ratings:.1f}%)")
print(f"  Matrix sparsity: {sparsity:.2f}%")

# ============================================================================
# 4. CREATE TRAIN/TEST SPLIT (80/20)
# ============================================================================
print("\n[4/6] Creating train/test split (80/20)...")

def decompose_matrix_80_20(R, random_seed=42):
    """Split rated entries into 80% train, 20% test"""
    R = np.array(R)
    A = np.zeros_like(R)  # Training set
    B = np.zeros_like(R)  # Test set

    np.random.seed(random_seed)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] == 1:
                if np.random.rand() < 0.8:
                    A[i, j] = 1
                else:
                    B[i, j] = 1

    return A, B

def replace_with_half(Y, R):
    """Set unrated entries to 0.5"""
    result = Y.copy()
    result[R == 0] = 0.5
    return result

R_train, R_test = decompose_matrix_80_20(R)
Y_train = replace_with_half(Y, R_train)
Y_test = replace_with_half(Y, R_test)

train_count = int(np.sum(R_train))
test_count = int(np.sum(R_test))

print(f"  Training set: {train_count:,} ratings ({100*train_count/total_ratings:.1f}%)")
print(f"  Test set: {test_count:,} ratings ({100*test_count/total_ratings:.1f}%)")

# ============================================================================
# 5. DEFINE COST FUNCTION
# ============================================================================

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Binary classification cost function for collaborative filtering.
    Uses sigmoid cross-entropy loss.
    """
    # Create mask: 1 for rated, 0 for unrated
    R_mask = tf.where(tf.equal(Y, 0.5),
                      tf.constant(0.0, dtype=tf.float32),
                      tf.constant(1.0, dtype=tf.float32))

    # Convert Y to float32
    Y_binary = tf.cast(Y, tf.float32)
    Y_binary = tf.where(tf.equal(R_mask, 1), Y_binary, tf.zeros_like(Y_binary))

    # Compute logits: X @ W^T + b
    logits = tf.matmul(X, W, transpose_b=True) + b

    # Binary cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_binary, logits=logits)

    # Apply mask to ignore unrated items
    masked_loss = loss * R_mask

    # Total cost with regularization
    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )

    return total_cost

def matrix_predict(X, W, b):
    """Predict binary ratings for all user-item pairs"""
    logits = tf.matmul(X, W, transpose_b=True) + b
    probabilities = tf.sigmoid(logits)
    binary_predictions = (probabilities >= 0.5).numpy().astype(int)
    return binary_predictions

def compute_accuracy(P, Y_test, R_test):
    """Compute accuracy on test set"""
    P = np.array(P)
    Y_test = np.array(Y_test)
    R_test = np.array(R_test)

    mask = (R_test == 1)
    P_masked = P[mask]
    Y_masked = Y_test[mask]

    matches = np.sum(P_masked == Y_masked)
    total = np.sum(mask)

    accuracy = matches / total if total > 0 else 0.0
    return accuracy

# ============================================================================
# 6. GRID SEARCH OVER FEATURES AND LAMBDA
# ============================================================================
print("\n[5/6] Running grid search...")
print("="*80)

results = []

for num_features in NUM_FEATURES_TO_TEST:
    for lambda_ in LAMBDA_VALUES:
        print(f"\nTesting: num_features={num_features}, lambda={lambda_}")

        # Initialize parameters
        tf.random.set_seed(42)
        W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float32), name='W')
        X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float32), name='X')
        b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float32), name='b')

        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # Training loop
        for iter in range(ITERATIONS):
            with tf.GradientTape() as tape:
                cost_value = cofi_cost_func_v(X, W, b, Y_train, R_train, lambda_)

            grads = tape.gradient(cost_value, [X, W, b])
            optimizer.apply_gradients(zip(grads, [X, W, b]))

            if iter % 50 == 0:
                print(f"  Iteration {iter}: loss = {cost_value:.1f}")

        # Final training loss
        final_train_loss = cofi_cost_func_v(X, W, b, Y_train, R_train, lambda_).numpy()

        # Evaluate on test set
        P = matrix_predict(X, W, b)
        test_accuracy = compute_accuracy(P, Y_test, R_test)

        print(f"  ✓ Final train loss: {final_train_loss:.1f}")
        print(f"  ✓ Test accuracy: {test_accuracy*100:.2f}%")

        results.append({
            'num_features': num_features,
            'lambda': lambda_,
            'train_loss': final_train_loss,
            'test_accuracy': test_accuracy
        })

# ============================================================================
# 7. REPORT BEST CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("[6/6] RESULTS SUMMARY")
print("="*80)

# Sort by test accuracy
results.sort(key=lambda x: x['test_accuracy'], reverse=True)

print("\nTop 5 Configurations (by test accuracy):\n")
print(f"{'Rank':<6} {'Features':<10} {'Lambda':<10} {'Test Acc':<12} {'Train Loss':<12}")
print("-"*80)

for i, result in enumerate(results[:5], 1):
    print(f"{i:<6} {result['num_features']:<10} {result['lambda']:<10} "
          f"{result['test_accuracy']*100:<11.2f}% {result['train_loss']:<12.1f}")

best = results[0]
print("\n" + "="*80)
print("RECOMMENDED CONFIGURATION:")
print(f"  Number of features: {best['num_features']}")
print(f"  Regularization (λ): {best['lambda']}")
print(f"  Expected test accuracy: {best['test_accuracy']*100:.2f}%")
print("="*80)

print("\nAll Results (sorted by test accuracy):\n")
for result in results:
    print(f"Features: {result['num_features']:2d}, Lambda: {result['lambda']:2d}, "
          f"Test Acc: {result['test_accuracy']*100:5.2f}%, Train Loss: {result['train_loss']:8.1f}")

print("\n✓ Optimization complete!")
print(f"  Tested {len(results)} configurations")
print(f"  Best accuracy: {best['test_accuracy']*100:.2f}%")
