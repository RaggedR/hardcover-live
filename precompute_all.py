#!/usr/bin/env python3
"""
Pre-compute All Recommendations

Run this locally to generate all recommendations, clusters, and user data.
The output files can then be deployed to Firebase with no ML dependencies.

Usage:
    python3 precompute_all.py

Output:
    webapp/data/recommendations.json - All pre-computed friend matches
    webapp/data/users.json - User list for dropdown
    webapp/data/clusters.json - Cluster information
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "webapp", "data")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 300
LEARNING_RATE = 0.1
NUM_CLUSTERS = 11

print("="*80)
print("PRE-COMPUTING ALL RECOMMENDATIONS")
print("="*80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA AND TRAIN MODEL
# ============================================================================
print("\n[1/4] Loading data and training model...")

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

# Filter
user_rating_counts = {
    u['user']['id']: sum(u['counts'].values())
    for u in users_data['user_books']
}

filtered_users = [
    u for u in users_data['user_books']
    if user_rating_counts[u['user']['id']] >= MIN_RATINGS_PER_USER
]

filtered_books = [
    b for b in books_data['books']
    if b['user_count'] >= MIN_USERS_PER_BOOK
]

user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(filtered_users)}
user_idx_to_id = {idx: u['user']['id'] for idx, u in enumerate(filtered_users)}
book_id_to_idx = {b['book']['id']: idx for idx, b in enumerate(filtered_books)}
book_idx_to_title = {idx: b['book']['title'] for idx, b in enumerate(filtered_books)}

num_movies = len(filtered_books)
num_users = len(filtered_users)

print(f"  {num_users} users, {num_movies} books")

# Build matrices
Y_raw = np.full((num_movies, num_users), np.nan)
R = np.zeros((num_movies, num_users))
user_books_dict = defaultdict(lambda: {'read': [], 'want': [], 'current': []})

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
        elif status_id == 2:
            Y_raw[book_idx, user_idx] = 0.7
            R[book_idx, user_idx] = 1
        elif status_id == 1:
            Y_raw[book_idx, user_idx] = 0.3
            R[book_idx, user_idx] = 1
        elif status_id == 5:
            Y_raw[book_idx, user_idx] = 0.0
            R[book_idx, user_idx] = 1

Y = np.nan_to_num(Y_raw, nan=0.5)

# Train model
print("  Training collaborative filtering...")

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
X = tf.Variable(tf.random.normal((num_movies, NUM_FEATURES), dtype=tf.float32) * 0.01, name='X')
b = tf.Variable(tf.zeros((1, num_users), dtype=tf.float32), name='b')

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

for iter in range(ITERATIONS):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Y, R, LAMBDA)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if (iter + 1) % 100 == 0:
        print(f"    Iteration {iter + 1}/{ITERATIONS}")

# ============================================================================
# 2. CLUSTER USERS
# ============================================================================
print("\n[2/4] Clustering users...")

user_features = W.numpy()
user_features_normalized = normalize(user_features, norm='l2', axis=1)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(user_features_normalized)

similarity_matrix = cosine_similarity(user_features_normalized)

print(f"  Created {NUM_CLUSTERS} clusters")

# ============================================================================
# 3. PRE-COMPUTE ALL RECOMMENDATIONS
# ============================================================================
print("\n[3/4] Pre-computing recommendations for all users...")

recommendations = {}

for user_idx in range(num_users):
    user_id = user_idx_to_id[user_idx]
    user = filtered_users[user_idx]['user']
    cluster_id = int(cluster_labels[user_idx])

    # Get cluster members
    cluster_user_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_users = [idx for idx in cluster_user_indices if idx != user_idx]

    # Calculate similarities and get matches
    matches = []
    for other_idx in cluster_users:
        other_user = filtered_users[other_idx]['user']
        similarity = float(similarity_matrix[user_idx, other_idx])

        # Find shared books
        user_read = set(user_books_dict[user_idx]['read'])
        other_read = set(user_books_dict[other_idx]['read'])
        shared_books = user_read & other_read
        shared_book_titles = [book_idx_to_title[idx] for idx in shared_books]

        # Books they've read that you want
        user_want = set(user_books_dict[user_idx]['want'])
        can_recommend = (other_read - user_read) & user_want
        recommend_titles = [book_idx_to_title[idx] for idx in can_recommend]

        matches.append({
            'user_id': int(user_idx_to_id[other_idx]),
            'name': other_user['name'],
            'username': other_user['username'],
            'similarity': round(similarity * 100, 1),  # Convert to percentage
            'shared_books': shared_book_titles[:10],  # Limit to top 10
            'can_recommend': recommend_titles[:5],  # Limit to top 5
            'num_read': len(other_read)
        })

    # Sort by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)

    # Store recommendations
    recommendations[str(user_id)] = {
        'user': {
            'id': int(user_id),
            'name': user['name'],
            'username': user['username']
        },
        'cluster_id': cluster_id,
        'cluster_name': f"Cluster {cluster_id}",
        'cluster_size': len(cluster_user_indices),
        'num_read': len(user_books_dict[user_idx]['read']),
        'num_want': len(user_books_dict[user_idx]['want']),
        'num_current': len(user_books_dict[user_idx]['current']),
        'matches': matches[:10]  # Top 10 matches
    }

    if (user_idx + 1) % 50 == 0:
        print(f"    Processed {user_idx + 1}/{num_users} users")

print(f"  ✓ Pre-computed recommendations for {num_users} users")

# ============================================================================
# 4. SAVE OUTPUT FILES
# ============================================================================
print("\n[4/4] Saving output files...")

# Save recommendations
recommendations_file = os.path.join(OUTPUT_DIR, "recommendations.json")
with open(recommendations_file, 'w') as f:
    json.dump(recommendations, f, indent=2)
print(f"  ✓ Saved recommendations.json ({os.path.getsize(recommendations_file) / 1024 / 1024:.1f}MB)")

# Save users list
users_list = [
    {
        'id': int(u['user']['id']),
        'name': u['user']['name'],
        'username': u['user']['username']
    }
    for u in filtered_users
]
users_list.sort(key=lambda x: x['name'].lower())

users_file = os.path.join(OUTPUT_DIR, "users.json")
with open(users_file, 'w') as f:
    json.dump(users_list, f, indent=2)
print(f"  ✓ Saved users.json ({len(users_list)} users)")

# Save cluster info
clusters_info = {}
for cluster_id in range(NUM_CLUSTERS):
    cluster_user_indices = np.where(cluster_labels == cluster_id)[0]
    clusters_info[str(cluster_id)] = {
        'id': cluster_id,
        'name': f"Cluster {cluster_id}",
        'size': len(cluster_user_indices),
        'members': [
            {
                'user_id': int(user_idx_to_id[idx]),
                'name': filtered_users[idx]['user']['name'],
                'username': filtered_users[idx]['user']['username']
            }
            for idx in cluster_user_indices
        ]
    }

clusters_file = os.path.join(OUTPUT_DIR, "clusters.json")
with open(clusters_file, 'w') as f:
    json.dump(clusters_info, f, indent=2)
print(f"  ✓ Saved clusters.json ({NUM_CLUSTERS} clusters)")

print("\n" + "="*80)
print("✓ PRE-COMPUTATION COMPLETE!")
print("="*80)
print(f"\nOutput files in: {OUTPUT_DIR}/")
print(f"  - recommendations.json")
print(f"  - users.json")
print(f"  - clusters.json")
print(f"\nNext steps:")
print(f"  1. Review the generated files")
print(f"  2. Deploy webapp/ directory to Firebase")
print("="*80)
