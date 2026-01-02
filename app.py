#!/usr/bin/env python3
"""
Book Friend Finder Web App

Simple Flask app for finding reading friends based on collaborative filtering
No password required - just select your name!
"""

from flask import Flask, render_template, request
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

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
NUM_CLUSTERS = 13

# Global variables for cached data
MODEL_LOADED = False
filtered_users = None
user_id_to_idx = None
user_idx_to_id = None
book_idx_to_title = None
user_books_dict = None
cluster_labels = None
cluster_names = None
similarity_matrix = None

def load_and_train_model():
    """Load data and train model once at startup"""
    global MODEL_LOADED, filtered_users, user_id_to_idx, user_idx_to_id
    global book_idx_to_title, user_books_dict, cluster_labels, cluster_names, similarity_matrix

    if MODEL_LOADED:
        return

    print("Loading data...")
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
    print("Training collaborative filtering...")

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

    print("Clustering users...")
    user_features = W.numpy()
    user_features_normalized = normalize(user_features, norm='l2', axis=1)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(user_features_normalized)

    # Name clusters
    cluster_names = {}
    for cluster_id in range(NUM_CLUSTERS):
        cluster_user_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_books = []
        for user_idx in cluster_user_indices:
            cluster_books.extend(user_books_dict[user_idx]['read'])

        book_counts = Counter(cluster_books)
        top_book_idx = book_counts.most_common(1)[0][0] if book_counts else 0
        top_book_title = book_idx_to_title[top_book_idx]

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

    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(user_features_normalized)

    MODEL_LOADED = True
    print("✓ Model ready!")

@app.route('/')
def index():
    """Main page - select user"""
    load_and_train_model()

    # Get sorted user list
    users_list = sorted(filtered_users, key=lambda u: u['user']['name'].lower())

    return render_template('index.html', users=users_list)

@app.route('/find_friends', methods=['POST'])
def find_friends():
    """Find friends for selected user"""
    load_and_train_model()

    user_id = int(request.form['user_id'])
    num_matches = int(request.form.get('num_matches', 5))

    if user_id not in user_id_to_idx:
        return "User not found", 404

    target_idx = user_id_to_idx[user_id]
    target_user = filtered_users[target_idx]
    target_cluster = cluster_labels[target_idx]

    # Get users in same cluster
    cluster_users = np.where(cluster_labels == target_cluster)[0]
    cluster_users = [u for u in cluster_users if u != target_idx]

    # User stats
    num_read = len(user_books_dict[target_idx]['read'])
    num_want = len(user_books_dict[target_idx]['want'])
    num_current = len(user_books_dict[target_idx]['current'])

    # Find matches
    matches = []
    for user_idx in cluster_users:
        similarity = similarity_matrix[target_idx, user_idx]

        # Find shared books
        target_read = set(user_books_dict[target_idx]['read'])
        match_read = set(user_books_dict[user_idx]['read'])
        shared_books = target_read & match_read

        # Get book titles
        shared_book_titles = [book_idx_to_title[idx] for idx in shared_books]

        # Find unique books they've read
        their_unique = match_read - target_read

        # Find books target wants that match has read
        target_want = set(user_books_dict[target_idx]['want'])
        can_recommend = their_unique & target_want
        recommend_titles = [book_idx_to_title[idx] for idx in can_recommend]

        matches.append({
            'user_idx': user_idx,
            'user_id': user_idx_to_id[user_idx],
            'name': filtered_users[user_idx]['user']['name'],
            'similarity': similarity,
            'shared_books': shared_book_titles,
            'can_recommend': recommend_titles,
            'num_read': len(match_read)
        })

    # Sort by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    top_matches = matches[:num_matches]

    return render_template('results.html',
                         user=target_user['user'],
                         num_read=num_read,
                         num_want=num_want,
                         num_current=num_current,
                         cluster_name=cluster_names[target_cluster],
                         cluster_id=target_cluster,
                         cluster_size=len(cluster_users) + 1,
                         matches=top_matches)

@app.route('/cluster/<int:cluster_id>')
def view_cluster(cluster_id):
    """View all members of a cluster with compatibility details"""
    load_and_train_model()

    if cluster_id < 0 or cluster_id >= NUM_CLUSTERS:
        return "Cluster not found", 404

    # Get viewing user from query parameter
    viewing_user_id = request.args.get('user_id', type=int)
    viewing_user_idx = None
    viewing_user = None

    if viewing_user_id and viewing_user_id in user_id_to_idx:
        viewing_user_idx = user_id_to_idx[viewing_user_id]
        viewing_user = filtered_users[viewing_user_idx]['user']

    # Get all users in this cluster
    cluster_user_indices = np.where(cluster_labels == cluster_id)[0]

    # Build member list with compatibility info
    members = []
    for user_idx in cluster_user_indices:
        user_entry = filtered_users[user_idx]
        num_read = len(user_books_dict[user_idx]['read'])
        num_want = len(user_books_dict[user_idx]['want'])
        num_current = len(user_books_dict[user_idx]['current'])

        member_data = {
            'user_id': user_entry['user']['id'],
            'name': user_entry['user']['name'],
            'username': user_entry['user']['username'],
            'num_read': num_read,
            'num_want': num_want,
            'num_current': num_current,
            'total_books': num_read + num_want + num_current,
            'is_viewing_user': viewing_user_idx == user_idx
        }

        # Calculate compatibility if viewing user is set
        if viewing_user_idx is not None and viewing_user_idx != user_idx:
            similarity = similarity_matrix[viewing_user_idx, user_idx]

            # Find shared books
            viewing_read = set(user_books_dict[viewing_user_idx]['read'])
            member_read = set(user_books_dict[user_idx]['read'])
            shared_books = viewing_read & member_read
            shared_book_titles = [book_idx_to_title[idx] for idx in shared_books]

            member_data['compatibility'] = similarity
            member_data['shared_books'] = shared_book_titles
            member_data['shared_count'] = len(shared_books)
        else:
            member_data['compatibility'] = None
            member_data['shared_books'] = []
            member_data['shared_count'] = 0

        members.append(member_data)

    # Sort by compatibility if viewing user exists, otherwise by total books
    if viewing_user_idx is not None:
        members.sort(key=lambda x: (not x['is_viewing_user'], -(x['compatibility'] or 0)), reverse=False)
    else:
        members.sort(key=lambda x: x['total_books'], reverse=True)

    return render_template('cluster.html',
                         cluster_id=cluster_id,
                         cluster_name=cluster_names[cluster_id],
                         members=members,
                         total_members=len(members),
                         viewing_user=viewing_user)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
