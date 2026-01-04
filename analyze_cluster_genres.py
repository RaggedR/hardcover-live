#!/usr/bin/env python3
"""
Analyze Cluster-Genre Correlation

Fetches genre data from Google Books API and analyzes whether
K-means clusters correlate with book genres.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import requests
import time
from collections import defaultdict, Counter
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Try to import optional libraries
try:
    from scipy.stats import chi2_contingency
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")
GENRE_CACHE_FILE = os.path.join(DATA_DIR, "book_genres_cache.json")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 300
LEARNING_RATE = 0.1
NUM_CLUSTERS = 13

TOP_N_BOOKS_PER_CLUSTER = 15
REQUEST_DELAY = 0.5


def load_genre_cache():
    """Load cached genre data"""
    try:
        with open(GENRE_CACHE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_genre_cache(cache):
    """Save genre data to cache"""
    with open(GENRE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_book_genres(title, author_name, cache):
    """Fetch genres from Google Books API with caching"""
    cache_key = f"{title}|{author_name}"

    if cache_key in cache:
        return cache[cache_key]

    print(f"    Fetching: {title} by {author_name}")

    try:
        query = f"{title} {author_name}".replace(" ", "+")
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"

        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        genres = []
        if 'items' in data and len(data['items']) > 0:
            volume_info = data['items'][0].get('volumeInfo', {})
            categories = volume_info.get('categories', [])

            for category in categories:
                # Split by "/" and take first part
                main_genre = category.split('/')[0].strip()
                genres.append(main_genre)

        if not genres:
            genres = ["Unknown"]

        cache[cache_key] = genres
        time.sleep(REQUEST_DELAY)

        return genres

    except Exception as e:
        print(f"      Error: {e}")
        cache[cache_key] = ["Unknown"]
        return ["Unknown"]


def main():
    print("="*80)
    print("CLUSTER-GENRE CORRELATION ANALYSIS")
    print("="*80)

    # Load data
    print("\n[1/5] Loading data...")
    with open(BOOKS_USERS_FILE, 'r') as f:
        books_data = json.load(f)

    with open(USER_BOOKS_FILE, 'r') as f:
        users_data = json.load(f)

    # Filter users and books (same as cluster_users.py)
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
    book_idx_to_data = {idx: b for idx, b in enumerate(filtered_books)}

    num_movies = len(filtered_books)
    num_users = len(filtered_users)

    print(f"  {num_users} users, {num_movies} books")

    # Build matrices
    print("\n[2/5] Building rating matrices...")
    Y_raw = np.full((num_movies, num_users), np.nan)
    R = np.zeros((num_movies, num_users))
    user_books_dict = defaultdict(lambda: defaultdict(list))

    for book_entry in filtered_books:
        book_idx = book_id_to_idx[book_entry['book']['id']]

        for user_entry in book_entry['users']:
            if user_entry['user_id'] not in user_id_to_idx:
                continue

            user_idx = user_id_to_idx[user_entry['user_id']]
            status_id = user_entry['status_id']
            rating = user_entry.get('rating')

            # Track books per user
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

    # Train collaborative filtering model
    print("\n[3/5] Training collaborative filtering model...")

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
            print(f"  Iteration {iter + 1}/{ITERATIONS}")

    # Cluster users
    print("\n[4/5] Clustering users...")
    user_features = W.numpy()
    user_features_normalized = normalize(user_features, norm='l2', axis=1)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(user_features_normalized)

    print(f"  Created {NUM_CLUSTERS} clusters")

    # Analyze genres per cluster
    print(f"\n[5/5] Analyzing genres per cluster...")

    genre_cache = load_genre_cache()
    print(f"  Loaded {len(genre_cache)} cached genres")

    cluster_genre_counts = defaultdict(Counter)
    cluster_book_counts = defaultdict(Counter)

    for cluster_id in range(NUM_CLUSTERS):
        print(f"\n  Cluster {cluster_id}:")

        # Get users in this cluster
        cluster_user_indices = np.where(cluster_labels == cluster_id)[0]

        # Get all books read by users in this cluster
        cluster_books = []
        for user_idx in cluster_user_indices:
            cluster_books.extend(user_books_dict[user_idx]['read'])

        # Count book frequencies
        book_counts = Counter(cluster_books)
        top_books = book_counts.most_common(TOP_N_BOOKS_PER_CLUSTER)

        print(f"    Analyzing top {len(top_books)} books...")

        # Get genres for top books
        for book_idx, count in top_books:
            book_data = book_idx_to_data[book_idx]
            book = book_data['book']
            title = book['title']

            author_name = "Unknown"
            if book.get('cached_contributors'):
                author_name = book['cached_contributors'][0]['author']['name']

            genres = get_book_genres(title, author_name, genre_cache)

            # Add to cluster genre counts
            for genre in genres:
                cluster_genre_counts[cluster_id][genre] += count

            cluster_book_counts[cluster_id][book_idx] = count

    # Save updated cache
    save_genre_cache(genre_cache)

    # Display results
    print("\n" + "="*80)
    print("RESULTS: GENRE DISTRIBUTION BY CLUSTER")
    print("="*80)

    all_genres = set()
    for cluster_id in range(NUM_CLUSTERS):
        all_genres.update(cluster_genre_counts[cluster_id].keys())

    all_genres = sorted(all_genres)

    for cluster_id in range(NUM_CLUSTERS):
        cluster_size = np.sum(cluster_labels == cluster_id)
        print(f"\nCluster {cluster_id} ({cluster_size} users):")
        print("-" * 60)

        total = sum(cluster_genre_counts[cluster_id].values())
        for genre, count in cluster_genre_counts[cluster_id].most_common(5):
            pct = (count / total * 100) if total > 0 else 0
            bar_length = int(pct / 2)
            bar = '█' * bar_length
            print(f"  {genre:25s} {bar} {count:3d} ({pct:5.1f}%)")

    # Statistical test
    if STATS_AVAILABLE and len(all_genres) > 1:
        print("\n" + "="*80)
        print("STATISTICAL TEST: Chi-Square Test of Independence")
        print("="*80)

        # Build contingency table
        contingency = []
        for cluster_id in range(NUM_CLUSTERS):
            row = [cluster_genre_counts[cluster_id][genre] for genre in all_genres]
            contingency.append(row)

        contingency = np.array(contingency)

        chi2, p_value, dof, expected = chi2_contingency(contingency)

        print(f"\nChi-square statistic: {chi2:.2f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Degrees of freedom: {dof}")

        if p_value < 0.05:
            print(f"\n✓ RESULT: Clusters DO significantly correlate with genres (p < 0.05)")
            print(f"  This suggests the latent features capture genre-related patterns.")
        else:
            print(f"\n✗ RESULT: Clusters do NOT significantly correlate with genres (p ≥ 0.05)")
            print(f"  The latent features may capture other reading patterns (frequency, diversity, etc.)")

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
