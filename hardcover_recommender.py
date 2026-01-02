#!/usr/bin/env python3
"""
Hardcover Book Recommender - Production System

Trains a collaborative filtering model using optimal parameters (5 features, λ=5)
and generates personalized book recommendations for users.

Usage:
  python3 hardcover_recommender.py                    # Recommend for random users
  python3 hardcover_recommender.py --user_id 62567    # Recommend for specific user
  python3 hardcover_recommender.py --top 20           # Show top 20 recommendations
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import argparse
from collections import defaultdict

# Configuration (optimal from grid search)
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_USERS_PER_BOOK = 2
NUM_FEATURES = 5  # Optimal from grid search
LAMBDA = 5        # Optimal from grid search
ITERATIONS = 200
LEARNING_RATE = 0.1

print("="*80)
print("HARDCOVER BOOK RECOMMENDER")
print("="*80)
print(f"Configuration: {NUM_FEATURES} features, λ={LAMBDA}")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

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
print(f"\n[2/5] Filtering to books with ≥{MIN_USERS_PER_BOOK} users...")

filtered_books = [
    book_entry for book_entry in books_data['books']
    if book_entry['user_count'] >= MIN_USERS_PER_BOOK
]

num_filtered_books = len(filtered_books)
print(f"  Kept {num_filtered_books}/{total_books} books ({100*num_filtered_books/total_books:.1f}%)")

# Create mappings
book_id_to_idx = {book_entry['book']['id']: idx for idx, book_entry in enumerate(filtered_books)}
book_idx_to_id = {idx: book_entry['book']['id'] for idx, book_entry in enumerate(filtered_books)}
book_idx_to_title = {idx: book_entry['book']['title'] for idx, book_entry in enumerate(filtered_books)}
book_idx_to_entry = {idx: book_entry for idx, book_entry in enumerate(filtered_books)}

user_id_to_idx = {user_entry['user']['id']: idx for idx, user_entry in enumerate(users_data['user_books'])}
user_idx_to_id = {idx: user_entry['user']['id'] for idx, user_entry in enumerate(users_data['user_books'])}
user_idx_to_name = {idx: user_entry['user']['name'] for idx, user_entry in enumerate(users_data['user_books'])}

num_movies = num_filtered_books
num_users = total_users

# ============================================================================
# 3. BUILD RATING MATRIX Y AND INDICATOR MATRIX R
# ============================================================================
print("\n[3/5] Building rating matrices...")

Y = np.full((num_movies, num_users), 0.5)
R = np.zeros((num_movies, num_users))

# Track what each user has already read
user_read_books = defaultdict(set)

for book_entry in filtered_books:
    book_idx = book_id_to_idx[book_entry['book']['id']]

    for user_entry in book_entry['users']:
        user_id = user_entry['user_id']
        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]
        status_id = user_entry['status_id']
        rating = user_entry.get('rating')

        # Track books user has interacted with
        if status_id in [1, 2, 3, 5]:
            user_read_books[user_idx].add(book_idx)

        # Binary rating logic
        if status_id == 3:  # Read
            if rating is not None:
                if rating >= 3:
                    Y[book_idx, user_idx] = 1
                    R[book_idx, user_idx] = 1
                else:
                    Y[book_idx, user_idx] = 0
                    R[book_idx, user_idx] = 1
            else:
                Y[book_idx, user_idx] = 1
                R[book_idx, user_idx] = 1
        elif status_id == 5:  # Did not finish
            Y[book_idx, user_idx] = 0
            R[book_idx, user_idx] = 1

total_ratings = int(np.sum(R))
print(f"  Total ratings: {total_ratings:,}")
print(f"  Matrix sparsity: {100 * (1 - total_ratings / (num_movies * num_users)):.2f}%")

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================
print(f"\n[4/5] Training model ({ITERATIONS} iterations)...")

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """Binary classification cost function"""
    R_mask = tf.where(tf.equal(Y, 0.5),
                      tf.constant(0.0, dtype=tf.float32),
                      tf.constant(1.0, dtype=tf.float32))

    Y_binary = tf.cast(Y, tf.float32)
    Y_binary = tf.where(tf.equal(R_mask, 1), Y_binary, tf.zeros_like(Y_binary))

    logits = tf.matmul(X, W, transpose_b=True) + b
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_binary, logits=logits)
    masked_loss = loss * R_mask

    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )

    return total_cost

# Initialize parameters
tf.random.set_seed(42)
W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), dtype=tf.float32), name='W')
X = tf.Variable(tf.random.normal((num_movies, NUM_FEATURES), dtype=tf.float32), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float32), name='b')

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Training loop
for iter in range(ITERATIONS):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Y, R, LAMBDA)

    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter % 50 == 0:
        print(f"  Iteration {iter}: loss = {cost_value:.1f}")

final_loss = cofi_cost_func_v(X, W, b, Y, R, LAMBDA).numpy()
print(f"  ✓ Training complete! Final loss: {final_loss:.1f}")

# ============================================================================
# 5. GENERATE RECOMMENDATIONS
# ============================================================================
print("\n[5/5] Generating recommendations...\n")
print("="*80)

def get_recommendations(user_idx, top_n=10, min_popularity=5):
    """
    Generate top N book recommendations for a user.

    Args:
        user_idx: Index of the user
        top_n: Number of recommendations to return
        min_popularity: Minimum number of users who rated the book

    Returns:
        List of (book_idx, title, probability, user_count) tuples
    """
    # Compute probabilities
    logits = tf.matmul(X, tf.expand_dims(W[user_idx], 1)) + b[0, user_idx]
    probabilities = tf.sigmoid(logits).numpy().flatten()

    # Get books user hasn't read
    read_books = user_read_books.get(user_idx, set())

    # Create recommendations list
    recommendations = []
    for book_idx in range(num_movies):
        # Skip books user has already read/interacted with
        if book_idx in read_books:
            continue

        book_entry = book_idx_to_entry[book_idx]
        user_count = book_entry['user_count']

        # Filter by popularity
        if user_count < min_popularity:
            continue

        recommendations.append({
            'book_idx': book_idx,
            'title': book_entry['book']['title'],
            'probability': probabilities[book_idx],
            'user_count': user_count,
            'authors': ', '.join([
                contrib['author']['name']
                for contrib in book_entry['book'].get('cached_contributors', [])
            ])
        })

    # Sort by probability
    recommendations.sort(key=lambda x: x['probability'], reverse=True)

    return recommendations[:top_n]

def display_recommendations(user_idx, top_n=10):
    """Display recommendations for a user"""
    user_id = user_idx_to_id[user_idx]
    user_name = user_idx_to_name[user_idx]

    print(f"RECOMMENDATIONS FOR: {user_name} (ID: {user_id})")
    print("="*80)

    # Show what they've read
    read_books = user_read_books.get(user_idx, set())
    print(f"\nBooks already read/in library: {len(read_books)}")

    # Get recommendations
    recs = get_recommendations(user_idx, top_n=top_n, min_popularity=5)

    print(f"\nTop {len(recs)} Recommended Books:\n")

    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['title']}")
        if rec['authors']:
            print(f"   by {rec['authors']}")
        print(f"   Probability: {rec['probability']:.1%} | Users: {rec['user_count']}")
        print()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate book recommendations')
parser.add_argument('--user_id', type=int, help='Specific user ID to recommend for')
parser.add_argument('--top', type=int, default=10, help='Number of recommendations (default: 10)')
parser.add_argument('--count', type=int, default=3, help='Number of users to show (default: 3)')
args = parser.parse_args()

# Generate recommendations
if args.user_id:
    # Specific user
    if args.user_id in user_id_to_idx:
        user_idx = user_id_to_idx[args.user_id]
        display_recommendations(user_idx, top_n=args.top)
    else:
        print(f"Error: User ID {args.user_id} not found!")
        print(f"Available user IDs range from {min(user_id_to_idx.keys())} to {max(user_id_to_idx.keys())}")
else:
    # Random sample of users
    for i, user_idx in enumerate(np.random.choice(num_users, min(args.count, num_users), replace=False)):
        if i > 0:
            print("\n" + "="*80 + "\n")
        display_recommendations(user_idx, top_n=args.top)

print("="*80)
print("✓ Recommendations complete!")
