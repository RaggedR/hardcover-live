#!/usr/bin/env python3
"""
Book Friend Finder - Simplified for Firebase Deployment

This version uses pre-computed recommendations (no TensorFlow needed).
All ML computation is done offline via precompute_all.py.
"""

from flask import Flask, render_template, request
import json
import os

app = Flask(__name__)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
CLUSTERS_FILE = os.path.join(DATA_DIR, "clusters.json")

# Load pre-computed data
print("Loading pre-computed data...")
try:
    with open(RECOMMENDATIONS_FILE, 'r') as f:
        RECOMMENDATIONS = json.load(f)
    print(f"✓ Loaded recommendations for {len(RECOMMENDATIONS)} users")
except FileNotFoundError:
    print("ERROR: recommendations.json not found!")
    print("Run: python3 precompute_all.py")
    RECOMMENDATIONS = {}

try:
    with open(USERS_FILE, 'r') as f:
        USERS = json.load(f)
    print(f"✓ Loaded {len(USERS)} users")
except FileNotFoundError:
    print("ERROR: users.json not found!")
    USERS = []

try:
    with open(CLUSTERS_FILE, 'r') as f:
        CLUSTERS = json.load(f)
    print(f"✓ Loaded {len(CLUSTERS)} clusters")
except FileNotFoundError:
    print("ERROR: clusters.json not found!")
    CLUSTERS = {}

print("✓ App ready!")


@app.route('/')
def index():
    """Main page - select user"""
    return render_template('index.html', users=USERS)


@app.route('/find_friends', methods=['POST'])
def find_friends():
    """Find friends for selected user"""
    user_id = request.form['user_id']

    # Look up pre-computed recommendations
    if user_id not in RECOMMENDATIONS:
        return f"No recommendations found for user {user_id}", 404

    data = RECOMMENDATIONS[user_id]

    return render_template('results.html',
                         user=data['user'],
                         num_read=data['num_read'],
                         num_want=data['num_want'],
                         num_current=data['num_current'],
                         cluster_name=data['cluster_name'],
                         cluster_id=data['cluster_id'],
                         cluster_size=data['cluster_size'],
                         matches=data['matches'])


@app.route('/cluster/<int:cluster_id>')
def view_cluster(cluster_id):
    """View all members of a cluster"""
    cluster_key = str(cluster_id)

    if cluster_key not in CLUSTERS:
        return "Cluster not found", 404

    # Get viewing user from query parameter
    viewing_user_id = request.args.get('user_id', type=int)
    viewing_user = None

    if viewing_user_id:
        viewing_user = next((u for u in USERS if u['id'] == viewing_user_id), None)

    cluster = CLUSTERS[cluster_key]
    members = cluster['members']

    # Add compatibility info if viewing user exists
    if viewing_user_id and str(viewing_user_id) in RECOMMENDATIONS:
        user_recs = RECOMMENDATIONS[str(viewing_user_id)]

        # Create a map of user_id -> similarity
        similarity_map = {
            m['user_id']: m['similarity']
            for m in user_recs['matches']
        }

        # Add compatibility to members
        for member in members:
            if member['user_id'] == viewing_user_id:
                member['is_viewing_user'] = True
                member['compatibility'] = None
            else:
                member['is_viewing_user'] = False
                member['compatibility'] = similarity_map.get(member['user_id'], 0)

        # Sort by compatibility
        members.sort(key=lambda x: (not x['is_viewing_user'], -(x['compatibility'] or 0)), reverse=False)
    else:
        for member in members:
            member['is_viewing_user'] = False
            member['compatibility'] = None

    return render_template('cluster.html',
                         cluster_id=cluster_id,
                         cluster_name=cluster['name'],
                         members=members,
                         total_members=len(members),
                         viewing_user=viewing_user)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
