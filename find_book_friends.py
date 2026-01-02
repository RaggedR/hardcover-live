#!/usr/bin/env python3
"""
Book Friend Finder - Friend "Dating" Site

Matches users based on reading habits and preferences.
Uses collaborative filtering embeddings + clustering to find:
- People in your reading cluster (similar tastes)
- Most compatible users (highest similarity scores)
- Shared books and conversation starters

Usage:
  python3 find_book_friends.py --user_id 62408
  python3 find_book_friends.py --user_id 62408 --matches 10
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import argparse
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

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
NUM_CLUSTERS = 13  # From clustering analysis

print("="*80)
print("📚 BOOK FRIEND FINDER - Find Your Reading Soulmate!")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
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

        # Track books for analysis
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
print(f"  Training collaborative filtering...")

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

print(f"  ✓ Model trained")

# ============================================================================
# 2. CLUSTER USERS
# ============================================================================
print(f"\n[2/4] Clustering users into {NUM_CLUSTERS} reading groups...")

user_features = W.numpy()
user_features_normalized = normalize(user_features, norm='l2', axis=1)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(user_features_normalized)

# Name clusters based on top books
cluster_names = {}
for cluster_id in range(NUM_CLUSTERS):
    cluster_user_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_books = []
    for user_idx in cluster_user_indices:
        cluster_books.extend(user_books_dict[user_idx]['read'])

    book_counts = Counter(cluster_books)
    top_book_idx = book_counts.most_common(1)[0][0] if book_counts else 0
    top_book_title = book_idx_to_title[top_book_idx]

    # Infer cluster type from top books
    if any(keyword in top_book_title.lower() for keyword in ['harry potter', 'hobbit', 'hunger games']):
        cluster_names[cluster_id] = "📚 Classic YA & Fantasy Fans"
    elif any(keyword in top_book_title.lower() for keyword in ['throne', 'court', 'crown']):
        cluster_names[cluster_id] = "👑 Romance Fantasy Readers"
    elif any(keyword in top_book_title.lower() for keyword in ['martian', 'three-body', 'foundation']):
        cluster_names[cluster_id] = "🚀 Sci-Fi Enthusiasts"
    elif any(keyword in top_book_title.lower() for keyword in ['normal people', 'pachinko', 'tomorrow']):
        cluster_names[cluster_id] = "📖 Literary Fiction Lovers"
    elif '1984' in top_book_title or 'animal farm' in top_book_title.lower():
        cluster_names[cluster_id] = "🎓 Classic Literature Readers"
    else:
        cluster_names[cluster_id] = f"📕 Group {cluster_id + 1}"

print(f"  ✓ Users grouped into reading clusters")

# ============================================================================
# 3. COMPUTE SIMILARITY MATRIX
# ============================================================================
print(f"\n[3/4] Computing compatibility scores (dot products!)...")

# Cosine similarity matrix (uses dot product of normalized vectors)
similarity_matrix = cosine_similarity(user_features_normalized)

print(f"  ✓ Compatibility matrix ready ({num_users}×{num_users})")

# ============================================================================
# 4. FRIEND MATCHING FUNCTION
# ============================================================================

def find_book_friends(target_user_id, num_matches=5):
    """Find best friend matches for a user"""

    if target_user_id not in user_id_to_idx:
        print(f"\n❌ Error: User {target_user_id} not found (needs ≥20 ratings)")
        print(f"\nAvailable users: {list(user_id_to_idx.keys())[:20]}...")
        return

    target_idx = user_id_to_idx[target_user_id]
    target_user = filtered_users[target_idx]
    target_cluster = cluster_labels[target_idx]
    target_name = target_user['user']['name']

    # Get users in same cluster
    cluster_users = np.where(cluster_labels == target_cluster)[0]
    cluster_users = [u for u in cluster_users if u != target_idx]  # Exclude self

    print(f"\n{'='*80}")
    print(f"🎯 FINDING BOOK FRIENDS FOR: {target_name}")
    print(f"{'='*80}")

    # User stats
    num_read = len(user_books_dict[target_idx]['read'])
    num_want = len(user_books_dict[target_idx]['want'])
    num_current = len(user_books_dict[target_idx]['current'])

    print(f"\n📊 Your Reading Profile:")
    print(f"  Books read: {num_read}")
    print(f"  Want to read: {num_want}")
    print(f"  Currently reading: {num_current}")
    print(f"  Reading cluster: {cluster_names[target_cluster]}")
    print(f"  Cluster size: {len(cluster_users) + 1} readers")

    # Get similarity scores for cluster members
    matches = []
    for user_idx in cluster_users:
        similarity = similarity_matrix[target_idx, user_idx]

        # Find shared books
        target_read = set(user_books_dict[target_idx]['read'])
        match_read = set(user_books_dict[user_idx]['read'])
        shared_books = target_read & match_read

        # Find unique books they've read
        their_unique = match_read - target_read

        # Find books target wants that match has read
        target_want = set(user_books_dict[target_idx]['want'])
        can_recommend = their_unique & target_want

        matches.append({
            'user_idx': user_idx,
            'user_id': user_idx_to_id[user_idx],
            'name': filtered_users[user_idx]['user']['name'],
            'similarity': similarity,
            'shared_books': shared_books,
            'can_recommend': can_recommend,
            'num_read': len(match_read)
        })

    # Sort by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)

    # Display top matches
    print(f"\n🤝 YOUR TOP {min(num_matches, len(matches))} BOOK FRIEND MATCHES:")
    print(f"{'='*80}\n")

    for i, match in enumerate(matches[:num_matches], 1):
        print(f"#{i} - {match['name']} (ID: {match['user_id']})")
        print(f"   ⭐ Compatibility: {match['similarity']:.1%}")
        print(f"   📚 Books read: {match['num_read']}")
        print(f"   🤝 Books in common: {len(match['shared_books'])}")

        # Show some shared books
        if match['shared_books']:
            shared_sample = list(match['shared_books'])[:5]
            print(f"\n   Shared favorites:")
            for book_idx in shared_sample:
                print(f"     • {book_idx_to_title[book_idx]}")

        # Show conversation starters
        if match['can_recommend']:
            print(f"\n   💬 Conversation starters:")
            print(f"     They've read {len(match['can_recommend'])} books you want to read!")
            rec_sample = list(match['can_recommend'])[:3]
            for book_idx in rec_sample:
                print(f"     • Ask them about: {book_idx_to_title[book_idx]}")

        print()

    # Also suggest a few from different clusters (for diversity)
    print(f"\n🌈 BONUS: Readers from Other Clusters (for variety)")
    print("-"*80)

    other_cluster_users = np.where(cluster_labels != target_cluster)[0]
    other_similarities = [(u, similarity_matrix[target_idx, u]) for u in other_cluster_users]
    other_similarities.sort(key=lambda x: x[1], reverse=True)

    for user_idx, sim in other_similarities[:3]:
        user = filtered_users[user_idx]
        user_cluster = cluster_labels[user_idx]
        print(f"  • {user['user']['name']} - {cluster_names[user_cluster]}")
        print(f"    Compatibility: {sim:.1%}")

    print(f"\n{'='*80}")
    print(f"💡 TIP: Higher compatibility = more similar reading tastes!")
    print(f"     Reach out and chat about your favorite books!")
    print(f"{'='*80}")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
print(f"\n[4/4] Ready to find book friends!")
print("="*80)

parser = argparse.ArgumentParser(description='Find book friends based on reading habits')
parser.add_argument('--user_id', type=int, required=True, help='User ID to find matches for')
parser.add_argument('--matches', type=int, default=5, help='Number of matches to show (default: 5)')
args = parser.parse_args()

find_book_friends(args.user_id, args.matches)

print(f"\n✓ Friend matching complete!")
print(f"\nTry with different users: python3 find_book_friends.py --user_id <ID>\n")
