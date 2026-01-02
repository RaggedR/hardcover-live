#!/usr/bin/env python3
"""
User Clustering by Reading Habits

Uses the user feature vectors (W matrix) from collaborative filtering
to cluster users into groups with similar reading preferences.

Determines optimal number of clusters using:
1. Elbow method (inertia)
2. Silhouette score
3. Calinski-Harabasz score

Then analyzes each cluster's favorite books and characteristics.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 300
LEARNING_RATE = 0.1

# Cluster range to test
MIN_CLUSTERS = 3
MAX_CLUSTERS = 15

print("="*80)
print("USER CLUSTERING BY READING HABITS")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND TRAIN MODEL (to get user embeddings)
# ============================================================================
print("\n[1/5] Loading data and training collaborative filtering...")

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
book_id_to_idx = {b['book']['id']: idx for idx, b in enumerate(filtered_books)}

num_movies = len(filtered_books)
num_users = len(filtered_users)

print(f"  {num_users} users, {num_movies} books")

# Build matrices with implicit feedback
Y_raw = np.full((num_movies, num_users), np.nan)
R = np.zeros((num_movies, num_users))
user_books_dict = defaultdict(lambda: defaultdict(list))  # For analysis later

for book_entry in filtered_books:
    book_idx = book_id_to_idx[book_entry['book']['id']]

    for user_entry in book_entry['users']:
        if user_entry['user_id'] not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_entry['user_id']]
        status_id = user_entry['status_id']
        rating = user_entry.get('rating')

        # Store for analysis
        if status_id == 3 and (rating is None or rating >= 3):
            user_books_dict[user_idx]['read'].append(book_idx)

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

# Train collaborative filtering to get user embeddings
print(f"  Training collaborative filtering ({ITERATIONS} iterations)...")

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

    if iter % 100 == 0:
        print(f"    Iteration {iter}: loss = {cost_value:.1f}")

print(f"  ✓ User embeddings ready (W matrix: {num_users} × {NUM_FEATURES})")

# ============================================================================
# 2. EXTRACT USER FEATURE VECTORS
# ============================================================================
print("\n[2/5] Extracting user feature vectors...")

user_features = W.numpy()  # Shape: (num_users, NUM_FEATURES)

# Option 1: L2 normalize for cosine similarity clustering
user_features_normalized = normalize(user_features, norm='l2', axis=1)

print(f"  Feature vectors: {user_features.shape}")
print(f"  Using normalized vectors for cosine-based clustering")

# ============================================================================
# 3. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
print(f"\n[3/5] Testing {MIN_CLUSTERS} to {MAX_CLUSTERS} clusters...")

results = []

for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
    # Use normalized features for better cosine-based clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(user_features_normalized)

    # Metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(user_features_normalized, cluster_labels)
    calinski = calinski_harabasz_score(user_features_normalized, cluster_labels)

    results.append({
        'k': k,
        'inertia': inertia,
        'silhouette': silhouette,
        'calinski': calinski,
        'model': kmeans
    })

    print(f"  K={k:2d}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}")

# Find optimal K based on metrics
best_silhouette = max(results, key=lambda x: x['silhouette'])
best_calinski = max(results, key=lambda x: x['calinski'])

print(f"\n  Best by Silhouette score: K={best_silhouette['k']} (score={best_silhouette['silhouette']:.3f})")
print(f"  Best by Calinski-Harabasz: K={best_calinski['k']} (score={best_calinski['calinski']:.1f})")

# Elbow method - find the "elbow" in inertia curve
print(f"\n  Elbow method (inertia decrease):")
for i in range(len(results) - 1):
    curr_inertia = results[i]['inertia']
    next_inertia = results[i + 1]['inertia']
    decrease_pct = (curr_inertia - next_inertia) / curr_inertia * 100
    print(f"    K={results[i]['k']} → K={results[i+1]['k']}: {decrease_pct:.1f}% decrease")

# Recommend optimal K (use silhouette as primary metric)
optimal_k = best_silhouette['k']
optimal_model = best_silhouette['model']

print(f"\n  ✓ RECOMMENDED: K={optimal_k} clusters")
print(f"    (Highest silhouette score - measures cluster separation)")

# ============================================================================
# 4. ANALYZE CLUSTERS
# ============================================================================
print(f"\n[4/5] Analyzing clusters (using K={optimal_k})...")

cluster_labels = optimal_model.labels_

# Analyze each cluster
cluster_info = []

for cluster_id in range(optimal_k):
    # Get users in this cluster
    cluster_user_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_size = len(cluster_user_indices)

    # Get all books read by users in this cluster
    cluster_books = []
    for user_idx in cluster_user_indices:
        cluster_books.extend(user_books_dict[user_idx]['read'])

    # Count book frequencies
    book_counts = Counter(cluster_books)
    most_common_books = book_counts.most_common(10)

    # Get book titles
    top_books = []
    for book_idx, count in most_common_books:
        book_entry = filtered_books[book_idx]
        top_books.append({
            'title': book_entry['book']['title'],
            'count': count,
            'percentage': count / cluster_size * 100
        })

    # Get sample users
    sample_users = []
    for user_idx in cluster_user_indices[:3]:
        user_entry = filtered_users[user_idx]
        sample_users.append(user_entry['user']['name'])

    cluster_info.append({
        'id': cluster_id,
        'size': cluster_size,
        'top_books': top_books,
        'sample_users': sample_users
    })

# ============================================================================
# 5. DISPLAY RESULTS
# ============================================================================
print(f"\n[5/5] Cluster Analysis Results")
print("="*80)

for cluster in cluster_info:
    print(f"\n📚 CLUSTER {cluster['id'] + 1} ({cluster['size']} users, {cluster['size']/num_users*100:.1f}%)")
    print("-"*80)

    print(f"\nTop Books in this cluster:")
    for i, book in enumerate(cluster['top_books'], 1):
        print(f"  {i}. {book['title']}")
        print(f"     {book['count']}/{cluster['size']} users ({book['percentage']:.1f}%)")

    print(f"\nSample users:")
    for user in cluster['sample_users']:
        print(f"  - {user}")

# Cluster size distribution
print("\n" + "="*80)
print("CLUSTER SIZE DISTRIBUTION")
print("="*80)

sorted_clusters = sorted(cluster_info, key=lambda x: x['size'], reverse=True)

for cluster in sorted_clusters:
    bar_length = int((cluster['size'] / num_users) * 50)
    bar = '█' * bar_length
    print(f"Cluster {cluster['id'] + 1}: {bar} {cluster['size']} users ({cluster['size']/num_users*100:.1f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nOptimal number of clusters: {optimal_k}")
print(f"Clustering method: K-Means with cosine similarity (L2-normalized features)")
print(f"Feature space: {NUM_FEATURES}-dimensional user embeddings from collaborative filtering")
print(f"\nSilhouette score: {best_silhouette['silhouette']:.3f}")
print(f"  (1.0 = perfect separation, 0 = overlapping clusters, <0 = wrong clusters)")
print(f"\nWhat the clusters represent:")
print(f"  - Users with similar reading preferences")
print(f"  - Each cluster has distinct favorite books")
print(f"  - Can be used for group recommendations or user segmentation")

print("\n" + "="*80)
print("✓ Clustering complete!")
print("\nNext steps:")
print("  - Adjust MIN_CLUSTERS and MAX_CLUSTERS to explore different K values")
print("  - Use clusters for: group recommendations, user segments, A/B testing")
print("="*80)
