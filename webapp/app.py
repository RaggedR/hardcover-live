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
BOOKS_FILE = os.path.join(DATA_DIR, "books.json")
BOOK_READERS_FILE = os.path.join(DATA_DIR, "book_readers.json")

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

try:
    with open(BOOKS_FILE, 'r') as f:
        BOOKS = json.load(f)
    print(f"✓ Loaded {len(BOOKS)} books")
except FileNotFoundError:
    print("WARNING: books.json not found (book club feature disabled)")
    BOOKS = []

try:
    with open(BOOK_READERS_FILE, 'r') as f:
        BOOK_READERS = json.load(f)
    print(f"✓ Loaded book readers data")
except FileNotFoundError:
    print("WARNING: book_readers.json not found (book club feature disabled)")
    BOOK_READERS = {}

print("✓ App ready!")

# Initialize chat database
chat_db.init_db()
print("✓ Chat database ready!")

# Create lookup dictionaries for users and books
USER_BY_ID = {u['id']: u for u in USERS}
BOOK_BY_ID = {b['id']: b for b in BOOKS}


@app.route('/')
def index():
    """Main page - select user"""
    return render_template('index.html', users=USERS)


@app.route('/find_friends', methods=['GET', 'POST'])
def find_friends():
    """Find friends for selected user"""
    if request.method == 'GET':
        return redirect(url_for('index'))
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


@app.route('/book_club')
def book_club():
    """Book club finder - search for a book to find compatible reading partners."""
    user_id = request.args.get('user_id', type=int)
    user = USER_BY_ID.get(user_id) if user_id else None

    # Get user's read books from recommendations
    user_books = []
    if user_id:
        user_key = str(user_id)
        if user_key in RECOMMENDATIONS:
            user_books = RECOMMENDATIONS[user_key].get('read_books', [])

    return render_template('book_club.html',
                         user=user,
                         user_books=user_books,
                         books=BOOKS,
                         total_books=len(BOOKS))


@app.route('/book_club/<int:book_id>')
def book_club_results(book_id):
    """Find compatible readers for a specific book."""
    user_id = request.args.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('book_club'))

    if book_id not in BOOK_BY_ID:
        return "Book not found", 404

    book = BOOK_BY_ID[book_id]
    user = USER_BY_ID[user_id]
    book_key = str(book_id)

    # Get all users who have read this book
    reader_ids = BOOK_READERS.get(book_key, [])

    # Filter to users in our system (excluding the current user)
    reader_ids = [rid for rid in reader_ids if rid in USER_BY_ID and rid != user_id]

    # Get similarity scores from recommendations
    user_key = str(user_id)
    similarity_map = {}
    shared_books_map = {}

    if user_key in RECOMMENDATIONS:
        for match in RECOMMENDATIONS[user_key]['matches']:
            similarity_map[match['user_id']] = match['similarity']
            shared_books_map[match['user_id']] = match.get('shared_books', [])

    # Build compatible readers list with similarity scores
    compatible_readers = []
    for rid in reader_ids:
        reader = USER_BY_ID[rid]
        similarity = similarity_map.get(rid, 0)
        shared_books = shared_books_map.get(rid, [])

        compatible_readers.append({
            'user_id': rid,
            'name': reader['name'],
            'username': reader['username'],
            'similarity': similarity,
            'shared_books': shared_books[:5]
        })

    # Sort by similarity (highest first)
    compatible_readers.sort(key=lambda x: x['similarity'], reverse=True)

    # Take top 10
    top_readers = compatible_readers[:10]

    return render_template('book_club_results.html',
                         user=user,
                         book=book,
                         readers=top_readers,
                         total_readers=len(reader_ids))


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

    # Check if allowed to chat: top 10 matches OR they've messaged you
    top_matches = get_top_matches(user_id, 10)
    can_chat = other_user_id in top_matches or chat_db.has_messaged_you(user_id, other_user_id)
    if not can_chat:
        return "You can only chat with your top 10 book friends or people who messaged you", 403

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


# ============================================================================
# GROUP CHAT (BOOK CLUB) ROUTES
# ============================================================================

@app.route('/group/<int:group_id>', methods=['GET', 'POST'])
def group_chat(group_id):
    """View or send messages in a group chat."""
    user_id = request.args.get('user_id', type=int) or request.form.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('index'))

    group = chat_db.get_group_by_id(group_id)
    if not group:
        return "Group not found", 404

    user = USER_BY_ID[user_id]

    # Auto-join if not a member (open book clubs)
    if not chat_db.is_group_member(group_id, user_id):
        chat_db.join_group_chat(group_id, user_id)

    # Handle POST (send message)
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if message:
            msg_id, timestamp = chat_db.send_group_message(group_id, user_id, message)
            return jsonify({
                'success': True,
                'timestamp': timestamp,
                'timestamp_formatted': format_timestamp(timestamp)
            })
        return jsonify({'success': False, 'error': 'Empty message'})

    # Get messages
    messages = chat_db.get_group_messages(group_id)
    for msg in messages:
        msg['timestamp_formatted'] = format_timestamp(msg['timestamp'])
        # Add user info
        if msg['from_user_id'] in USER_BY_ID:
            msg['from_user'] = USER_BY_ID[msg['from_user_id']]
        else:
            msg['from_user'] = {'name': f"User {msg['from_user_id']}", 'username': 'unknown'}

    last_timestamp = messages[-1]['timestamp'] if messages else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get members
    members = chat_db.get_group_members(group_id)
    member_info = []
    for m in members:
        if m['user_id'] in USER_BY_ID:
            member_info.append(USER_BY_ID[m['user_id']])

    # Get book info if available
    book = BOOK_BY_ID.get(group['book_id'])

    return render_template('group_chat.html',
                         user=user,
                         group=group,
                         book=book,
                         messages=messages,
                         members=member_info,
                         last_timestamp=last_timestamp)


@app.route('/group/<int:group_id>/messages')
def group_poll_messages(group_id):
    """Poll for new group messages (JSON endpoint)."""
    user_id = request.args.get('user_id', type=int)
    since = request.args.get('since', '')

    if not user_id or user_id not in USER_BY_ID:
        return jsonify({'messages': []})

    if not chat_db.is_group_member(group_id, user_id):
        return jsonify({'messages': []})

    messages = chat_db.get_group_messages(group_id, since=since if since else None)
    for msg in messages:
        msg['timestamp_formatted'] = format_timestamp(msg['timestamp'])
        if msg['from_user_id'] in USER_BY_ID:
            msg['from_user'] = USER_BY_ID[msg['from_user_id']]
        else:
            msg['from_user'] = {'name': f"User {msg['from_user_id']}", 'username': 'unknown'}

    return jsonify({'messages': messages})


@app.route('/group/create/<int:book_id>', methods=['POST'])
def create_book_club(book_id):
    """Create a new book club group chat."""
    user_id = request.form.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('index'))

    if book_id not in BOOK_BY_ID:
        return "Book not found", 404

    book = BOOK_BY_ID[book_id]

    # Check if group already exists for this book
    existing = chat_db.get_group_chat_for_book(book_id)
    if existing:
        return redirect(url_for('group_chat', group_id=existing['id'], user_id=user_id))

    # Create new group
    group_id = chat_db.create_group_chat(book_id, book['title'], user_id)

    return redirect(url_for('group_chat', group_id=group_id, user_id=user_id))


@app.route('/my_groups')
def my_groups():
    """View all book club groups the user is a member of."""
    user_id = request.args.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('index'))

    user = USER_BY_ID[user_id]
    groups = chat_db.get_user_groups(user_id)

    # Add book info to each group
    for g in groups:
        g['book'] = BOOK_BY_ID.get(g['book_id'])

    return render_template('my_groups.html', user=user, groups=groups)


@app.route('/invite', methods=['POST'])
def send_invite():
    """Send a book club invitation to another user."""
    from_user_id = request.form.get('from_user_id', type=int)
    to_user_id = request.form.get('to_user_id', type=int)
    book_id = request.form.get('book_id', type=int)

    if not all([from_user_id, to_user_id, book_id]):
        return jsonify({'success': False, 'error': 'Missing parameters'})

    if from_user_id not in USER_BY_ID or to_user_id not in USER_BY_ID:
        return jsonify({'success': False, 'error': 'User not found'})

    if book_id not in BOOK_BY_ID:
        return jsonify({'success': False, 'error': 'Book not found'})

    book = BOOK_BY_ID[book_id]
    from_user = USER_BY_ID[from_user_id]

    # Check if group exists for this book, create if not
    group = chat_db.get_group_chat_for_book(book_id)
    if not group:
        group_id = chat_db.create_group_chat(book_id, book['title'], from_user_id)
    else:
        group_id = group['id']

    # Send invitation as a regular message with full URL
    base_url = request.host_url.rstrip('/')
    group_link = f"{base_url}/group/{group_id}?user_id={to_user_id}"
    invite_message = f"📚 Book Club Invitation!\n\nI'd love to discuss \"{book['title']}\" with you!\n\nJoin the discussion: {group_link}"
    msg_id, timestamp = chat_db.send_message(from_user_id, to_user_id, invite_message)

    return jsonify({'success': True, 'message_id': msg_id})


@app.route('/invitations')
def view_invitations():
    """View all invitations for the current user."""
    user_id = request.args.get('user_id', type=int)
    if not user_id or user_id not in USER_BY_ID:
        return redirect(url_for('index'))

    user = USER_BY_ID[user_id]
    invitations = chat_db.get_user_invitations(user_id)

    # Mark all as seen
    chat_db.mark_all_invitations_seen(user_id)

    # Enrich with user info
    for inv in invitations:
        if inv['from_user_id'] in USER_BY_ID:
            inv['from_user'] = USER_BY_ID[inv['from_user_id']]
        else:
            inv['from_user'] = {'name': f"User {inv['from_user_id']}", 'username': 'unknown'}
        inv['book'] = BOOK_BY_ID.get(inv['book_id'])

    return render_template('invitations.html', user=user, invitations=invitations)


@app.route('/invitations/count')
def invitation_count():
    """Get unseen invitation count."""
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({'count': 0})

    count = chat_db.get_unseen_invitation_count(user_id)
    return jsonify({'count': count})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))
    app.run(debug=True, host='0.0.0.0', port=port)
