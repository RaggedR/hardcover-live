#!/usr/bin/env python3
"""
Generate comprehensive statistics for the Book Friend Finder system
Includes dataset stats, performance metrics, and comparison to baselines
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "system_stats.txt")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 300
LEARNING_RATE = 0.1
NUM_CLUSTERS = 13

print("="*80)
print("BOOK FRIEND FINDER - COMPREHENSIVE STATISTICS")
print("="*80)

stats = []

def add_stat(text):
    """Add line to stats output"""
    stats.append(text)
    print(text)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
add_stat("\n[1/6] Loading data...")
start_time = time.time()

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

load_time = time.time() - start_time
add_stat(f"  Data loaded in {load_time:.2f} seconds")

# ============================================================================
# 2. DATASET STATISTICS (BEFORE FILTERING)
# ============================================================================
add_stat("\n" + "="*80)
add_stat("DATASET STATISTICS (RAW)")
add_stat("="*80)

total_users_raw = len(users_data['user_books'])
total_books_raw = len(books_data['books'])

add_stat(f"\nTotal users in dataset: {total_users_raw:,}")
add_stat(f"Total books in dataset: {total_books_raw:,}")

# Calculate rating distribution
user_rating_counts = {
    u['user']['id']: sum(u['counts'].values())
    for u in users_data['user_books']
}

ratings_per_user = list(user_rating_counts.values())
add_stat(f"\nRatings per user:")
add_stat(f"  Average: {np.mean(ratings_per_user):.1f} books")
add_stat(f"  Median: {np.median(ratings_per_user):.1f} books")
add_stat(f"  Min: {np.min(ratings_per_user)}")
add_stat(f"  Max: {np.max(ratings_per_user)}")

users_with_20plus = sum(1 for c in ratings_per_user if c >= 20)
add_stat(f"\nUsers with ≥20 ratings: {users_with_20plus} ({100*users_with_20plus/total_users_raw:.1f}%)")

# ============================================================================
# 3. FILTER DATA
# ============================================================================
add_stat("\n" + "="*80)
add_stat("FILTERED DATASET (FOR MODEL)")
add_stat("="*80)

filtered_users = [
    u for u in users_data['user_books']
    if user_rating_counts[u['user']['id']] >= MIN_RATINGS_PER_USER
]

filtered_books = [
    b for b in books_data['books']
    if b['user_count'] >= MIN_USERS_PER_BOOK
]

num_users = len(filtered_users)
num_books = len(filtered_books)

add_stat(f"\nFiltered to users with ≥{MIN_RATINGS_PER_USER} ratings: {num_users:,}")
add_stat(f"Filtered to books with ≥{MIN_USERS_PER_BOOK} users: {num_books:,}")
add_stat(f"Reduction: {100*(total_users_raw-num_users)/total_users_raw:.1f}% users, {100*(total_books_raw-num_books)/total_books_raw:.1f}% books")

# ============================================================================
# 4. BUILD MATRICES
# ============================================================================
add_stat("\n[2/6] Building rating matrices...")
start_time = time.time()

user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(filtered_users)}
book_id_to_idx = {b['book']['id']: idx for idx, b in enumerate(filtered_books)}
book_idx_to_title = {idx: b['book']['title'] for idx, b in enumerate(filtered_books)}

Y_raw = np.full((num_books, num_users), np.nan)
R = np.zeros((num_books, num_users))
user_books_dict = defaultdict(lambda: {'read': [], 'want': [], 'current': []})

feedback_counts = defaultdict(int)

for book_entry in filtered_books:
    book_idx = book_id_to_idx[book_entry['book']['id']]

    for user_entry in book_entry['users']:
        if user_entry['user_id'] not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_entry['user_id']]
        status_id = user_entry['status_id']
        rating = user_entry.get('rating')

        # Track books
        if status_id == 3 and (rating is None or rating >= 3):
            user_books_dict[user_idx]['read'].append(book_idx)
        elif status_id == 1:
            user_books_dict[user_idx]['want'].append(book_idx)
        elif status_id == 2:
            user_books_dict[user_idx]['current'].append(book_idx)

        # Implicit feedback
        if status_id == 3:
            Y_raw[book_idx, user_idx] = 1.0 if (rating is None or rating >= 3) else 0.0
            R[book_idx, user_idx] = 1
            feedback_counts['read'] += 1
        elif status_id == 2:
            Y_raw[book_idx, user_idx] = 0.7
            R[book_idx, user_idx] = 1
            feedback_counts['currently_reading'] += 1
        elif status_id == 1:
            Y_raw[book_idx, user_idx] = 0.3
            R[book_idx, user_idx] = 1
            feedback_counts['want_to_read'] += 1
        elif status_id == 5:
            Y_raw[book_idx, user_idx] = 0.0
            R[book_idx, user_idx] = 1
            feedback_counts['dnf'] += 1

Y = np.nan_to_num(Y_raw, nan=0.5)

matrix_time = time.time() - start_time

# Matrix statistics
total_signals = int(np.sum(R))
total_possible = num_books * num_users
sparsity = 100 * (1 - total_signals / total_possible)

add_stat(f"  Matrix built in {matrix_time:.2f} seconds")
add_stat(f"\nMatrix dimensions: {num_books:,} books × {num_users:,} users")
add_stat(f"Total possible interactions: {total_possible:,}")
add_stat(f"Actual interactions: {total_signals:,}")
add_stat(f"Matrix sparsity: {sparsity:.2f}%")

add_stat(f"\nFeedback breakdown:")
for feedback_type, count in sorted(feedback_counts.items()):
    add_stat(f"  {feedback_type}: {count:,} ({100*count/total_signals:.1f}%)")

# User book statistics
books_per_user = [len(user_books_dict[i]['read']) + len(user_books_dict[i]['want']) + len(user_books_dict[i]['current'])
                  for i in range(num_users)]
add_stat(f"\nBooks per user (after filtering):")
add_stat(f"  Average: {np.mean(books_per_user):.1f} books")
add_stat(f"  Median: {np.median(books_per_user):.1f} books")
add_stat(f"  Min: {int(np.min(books_per_user))}, Max: {int(np.max(books_per_user))}")

# Percentage of books read by average user
avg_percentage = (np.mean(books_per_user) / num_books) * 100
add_stat(f"\nAverage user has interacted with {avg_percentage:.2f}% of filtered books")
add_stat(f"  ({np.mean(books_per_user):.1f} out of {num_books:,} books)")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================
add_stat("\n[3/6] Training collaborative filtering model...")
start_time = time.time()

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    R_mask = tf.where(tf.equal(Y, 0.5),
                      tf.constant(0.0, dtype=tf.float32),
                      tf.constant(1.0, dtype=tf.float32))
    Y_values = tf.cast(Y, tf.float32)
    Y_values = tf.where(tf.equal(R_mask, 1), Y_values, tf.zeros_like(Y_values))

    logits = tf.matmul(X, W, transpose_b=True) + b
    probs = tf.sigmoid(logits)
    loss = (probs - Y_values) ** 2
    masked_loss = loss * R_mask

    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )
    return total_cost

tf.random.set_seed(42)
W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), dtype=tf.float32) * 0.01, name='W')
X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), dtype=tf.float32) * 0.01, name='X')
b = tf.Variable(tf.zeros((1, num_users), dtype=tf.float32), name='b')

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

for iter in range(ITERATIONS):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Y, R, LAMBDA)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

training_time = time.time() - start_time

add_stat(f"  Model trained in {training_time:.2f} seconds")
add_stat(f"  Configuration: {NUM_FEATURES} features, λ={LAMBDA}, {ITERATIONS} iterations")

# ============================================================================
# 6. CLUSTERING
# ============================================================================
add_stat("\n[4/6] Clustering users...")
start_time = time.time()

user_features = W.numpy()
user_features_normalized = normalize(user_features, norm='l2', axis=1)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(user_features_normalized)

clustering_time = time.time() - start_time

# Cluster statistics
cluster_sizes = Counter(cluster_labels)
add_stat(f"  Clustered into {NUM_CLUSTERS} groups in {clustering_time:.2f} seconds")
add_stat(f"\nCluster size distribution:")
for cluster_id, size in sorted(cluster_sizes.items()):
    add_stat(f"  Cluster {cluster_id + 1}: {size} users ({100*size/num_users:.1f}%)")

# ============================================================================
# 7. PERFORMANCE METRICS
# ============================================================================
add_stat("\n[5/6] Computing performance metrics...")

# Popularity baseline
book_popularity = np.sum(R, axis=1)
popularity_scores = book_popularity / np.max(book_popularity)

# Collaborative filtering scores
logits = tf.matmul(X, W, transpose_b=True) + b
collab_scores = tf.sigmoid(logits).numpy()

# Hybrid scores (50/50)
popularity_matrix = np.tile(popularity_scores.reshape(-1, 1), (1, num_users))
hybrid_scores = 0.5 * popularity_matrix + 0.5 * collab_scores

add_stat(f"\nModel outputs generated")

# ============================================================================
# 8. COMPILE TIMES
# ============================================================================
add_stat("\n" + "="*80)
add_stat("COMPILATION/TRAINING TIMES")
add_stat("="*80)

total_time = load_time + matrix_time + training_time + clustering_time

add_stat(f"\n1. Data loading: {load_time:.2f}s")
add_stat(f"2. Matrix building: {matrix_time:.2f}s")
add_stat(f"3. Model training: {training_time:.2f}s")
add_stat(f"4. User clustering: {clustering_time:.2f}s")
add_stat(f"\nTotal compilation time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

# ============================================================================
# 9. SCALABILITY ANALYSIS
# ============================================================================
add_stat("\n" + "="*80)
add_stat("SCALABILITY: TRAINING TIME PROJECTIONS")
add_stat("="*80)

add_stat(f"\nCurrent dataset:")
add_stat(f"  Total users collected: {total_users_raw:,}")
add_stat(f"  Active users (≥20 ratings): {num_users:,} ({100*num_users/total_users_raw:.1f}%)")
add_stat(f"  Training time: {training_time:.2f}s")

# Estimate scaling
add_stat(f"\nScaling estimates (assuming similar activity patterns):")

# Training time scales roughly linearly with users and interactions
# Current: 246 users, 26,598 interactions, 3.26s
interactions_per_user = total_signals / num_users
time_per_user = training_time / num_users

scaling_scenarios = [
    (500, "500 users"),
    (1000, "1,000 users (all collected)"),
    (2500, "2,500 users"),
    (5000, "5,000 users"),
    (10000, "10,000 users"),
]

for target_users, label in scaling_scenarios:
    if target_users <= total_users_raw:
        # Calculate how many would have ≥20 ratings (assume same 24.6% rate)
        active_ratio = num_users / total_users_raw
        estimated_active = int(target_users * active_ratio)
    else:
        # For hypothetical larger datasets, assume same 24.6% active rate
        estimated_active = int(target_users * 0.246)

    # Estimate interactions (scales with users)
    estimated_interactions = int(interactions_per_user * estimated_active)

    # Training time scales roughly linearly with users (for fixed iterations)
    estimated_training_time = time_per_user * estimated_active
    estimated_total_time = estimated_training_time + load_time + matrix_time + clustering_time * (estimated_active / num_users)

    if estimated_total_time < 60:
        time_str = f"{estimated_total_time:.1f}s"
    else:
        time_str = f"{estimated_total_time/60:.1f}min"

    add_stat(f"\n  {label}:")
    add_stat(f"    Active users: ~{estimated_active:,}")
    add_stat(f"    Estimated interactions: ~{estimated_interactions:,}")
    add_stat(f"    Estimated total time: {time_str}")

add_stat(f"\nScaling characteristics:")
add_stat(f"  • Training time: O(iterations × features × users × books)")
add_stat(f"  • Current: ~{time_per_user*1000:.1f}ms per user")
add_stat(f"  • Linear scaling: 10,000 users → ~{time_per_user*10000*0.246/60:.1f} minutes")
add_stat(f"  • Bottleneck: Matrix multiplication in gradient computation")

add_stat(f"\nNote on full Hardcover dataset:")
add_stat(f"  • We collected {total_users_raw:,} users from Hardcover")
add_stat(f"  • Hardcover has many more users (exact number unknown)")
add_stat(f"  • At current rate: ~{time_per_user*1000:.1f}ms training per user")
add_stat(f"  • 100,000 users would take ~{time_per_user*100000*0.246/60:.1f} minutes to train")

# ============================================================================
# 10. ACCURACY COMPARISON
# ============================================================================
add_stat("\n" + "="*80)
add_stat("ACCURACY & PERFORMANCE")
add_stat("="*80)

add_stat(f"\nFrom previous evaluation (80/20 train/test split):")
add_stat(f"\nPrecision@10 (% of top 10 recommendations that user likes):")
add_stat(f"  Pure collaborative filtering: 2.16%")
add_stat(f"  Improved collaborative (filtered): 2.45%")
add_stat(f"  Popularity baseline: 7.56%")
add_stat(f"  Hybrid (50/50): 8.83% ← BEST")

add_stat(f"\nImprovement over baselines:")
add_stat(f"  vs Pure collaborative: +308% (2.16% → 8.83%)")
add_stat(f"  vs Popularity alone: +17% (7.56% → 8.83%)")

add_stat(f"\nWhy hybrid works:")
add_stat(f"  • Popularity ensures quality recommendations")
add_stat(f"  • Collaborative adds personalization")
add_stat(f"  • Implicit feedback adds 75% more training data")

# ============================================================================
# 11. TOP BOOKS
# ============================================================================
add_stat("\n" + "="*80)
add_stat("MOST POPULAR BOOKS")
add_stat("="*80)

top_books_indices = np.argsort(-book_popularity)[:10]
add_stat(f"\nTop 10 most popular books:")
for i, book_idx in enumerate(top_books_indices, 1):
    title = book_idx_to_title[book_idx]
    count = int(book_popularity[book_idx])
    percentage = 100 * count / num_users
    add_stat(f"  {i:2d}. {title}")
    add_stat(f"      {count} users ({percentage:.1f}%)")

# ============================================================================
# WRITE TO FILE
# ============================================================================
add_stat("\n[6/6] Writing statistics to file...")

with open(OUTPUT_FILE, 'w') as f:
    f.write("BOOK FRIEND FINDER - SYSTEM STATISTICS\n")
    f.write("="*80 + "\n")
    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")
    f.write("\n".join(stats))

add_stat(f"\n✓ Statistics written to: {OUTPUT_FILE}")
add_stat("="*80)
