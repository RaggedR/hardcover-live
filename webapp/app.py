#!/usr/bin/env python3
"""
Book Friend Finder - Simplified for Firebase Deployment

This version uses pre-computed recommendations (no TensorFlow needed).
All ML computation is done offline via precompute_all.py.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os
from datetime import datetime
import chat_db

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

# Initialize chat database
chat_db.init_db()
print("✓ Chat database ready!")

# Create lookup dictionaries for users
USER_BY_ID = {u['id']: u for u in USERS}


@app.route('/')
def index():
    """Main page - select user"""
    return render_template('index.html', users=USERS)


@app.route('/find_friends', methods=['POST'])
def find_friends():
    """Find friends for selected user"""
    user_id = request.form['user_id']
    num_matches = int(request.form.get('num_matches', 5))

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
                         matches=data['matches'][:num_matches])


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


def get_top_matches(user_id: int, num_matches: int = 10) -> list:
    """Get top N matches for a user from pre-computed data. Returns list of user_ids."""
    user_key = str(user_id)
    if user_key not in RECOMMENDATIONS:
        return []

    matches = RECOMMENDATIONS[user_key].get('matches', [])
    return [m['user_id'] for m in matches[:num_matches]]


def format_timestamp(ts_str: str) -> str:
    """Format a timestamp string for display."""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        if dt.date() == now.date():
            return dt.strftime("%I:%M %p")
        else:
            return dt.strftime("%b %d, %I:%M %p")
    except:
        return ts_str


@app.route('/chat')
def chat_inbox():
    """View chat inbox with all conversations."""
    user_id = request.args.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('index'))

    user = USER_BY_ID[user_id]

    # Get conversations
    conversations = chat_db.get_inbox(user_id)

    # Enrich with user names
    for conv in conversations:
        other_id = conv['other_user_id']
        if other_id in USER_BY_ID:
            conv['other_user_name'] = USER_BY_ID[other_id]['name']
        else:
            conv['other_user_name'] = f"User {other_id}"
        conv['last_timestamp_formatted'] = format_timestamp(conv['last_timestamp'])

    return render_template('chat_inbox.html',
                         user=user,
                         conversations=conversations)


@app.route('/chat/unread_count')
def chat_unread_count():
    """Get unread message count for polling."""
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({'count': 0})

    count = chat_db.get_unread_count(user_id)
    return jsonify({'count': count})


@app.route('/chat/<int:other_user_id>', methods=['GET', 'POST'])
def chat_conversation(other_user_id):
    """View or send messages in a conversation."""
    user_id = request.args.get('user_id', type=int) or request.form.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('index'))

    if other_user_id not in USER_BY_ID:
        return "User not found", 404

    # Check if other_user is in top 10 matches (access control)
    top_matches = get_top_matches(user_id, 10)
    if other_user_id not in top_matches:
        return "You can only chat with your top 10 book friends", 403

    user = USER_BY_ID[user_id]
    other_user = USER_BY_ID[other_user_id]

    # Handle POST (send message)
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if message:
            msg_id, timestamp = chat_db.send_message(user_id, other_user_id, message)
            return jsonify({
                'success': True,
                'timestamp': timestamp,
                'timestamp_formatted': format_timestamp(timestamp)
            })
        return jsonify({'success': False, 'error': 'Empty message'})

    # Mark messages as read
    chat_db.mark_as_read(user_id, other_user_id)

    # Get conversation
    messages = chat_db.get_conversation(user_id, other_user_id)
    for msg in messages:
        msg['timestamp_formatted'] = format_timestamp(msg['timestamp'])

    last_timestamp = messages[-1]['timestamp'] if messages else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get compatibility from pre-computed data
    user_key = str(user_id)
    compatibility = None
    shared_books = []
    if user_key in RECOMMENDATIONS:
        for match in RECOMMENDATIONS[user_key]['matches']:
            if match['user_id'] == other_user_id:
                compatibility = match.get('similarity', 0)
                shared_books = match.get('shared_books', [])[:5]
                break

    return render_template('chat_conversation.html',
                         user=user,
                         other_user=other_user,
                         messages=messages,
                         last_timestamp=last_timestamp,
                         compatibility=compatibility,
                         shared_books=shared_books)


@app.route('/chat/<int:other_user_id>/messages')
def chat_poll_messages(other_user_id):
    """Poll for new messages (JSON endpoint)."""
    user_id = request.args.get('user_id', type=int)
    since = request.args.get('since', '')

    if not user_id or user_id not in USER_BY_ID:
        return jsonify({'messages': []})

    if other_user_id not in USER_BY_ID:
        return jsonify({'messages': []})

    # Mark messages as read
    chat_db.mark_as_read(user_id, other_user_id)

    # Get new messages
    messages = chat_db.get_conversation(user_id, other_user_id, since=since if since else None)
    for msg in messages:
        msg['timestamp_formatted'] = format_timestamp(msg['timestamp'])

    return jsonify({'messages': messages})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    app.run(debug=True, host='0.0.0.0', port=port)
