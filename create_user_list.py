#!/usr/bin/env python3
"""
Create a text file with list of active users (≥20 ratings)
"""

import json
import os

DATA_DIR = os.path.expanduser("~/data/hardcover/")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")
MIN_RATINGS_PER_USER = 20

print("Loading user data...")
with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

print("Filtering active users...")
user_rating_counts = {
    u['user']['id']: sum(u['counts'].values())
    for u in users_data['user_books']
}

filtered_users = [
    u for u in users_data['user_books']
    if user_rating_counts[u['user']['id']] >= MIN_RATINGS_PER_USER
]

# Sort by name
filtered_users.sort(key=lambda u: u['user']['name'].lower())

# Write to file
output_file = os.path.join(DATA_DIR, "active_users.txt")
with open(output_file, 'w') as f:
    f.write("HARDCOVER BOOK FRIENDS - ACTIVE USERS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Users with ≥{MIN_RATINGS_PER_USER} ratings: {len(filtered_users)}\n")
    f.write("=" * 60 + "\n\n")

    for user in filtered_users:
        user_id = user['user']['id']
        name = user['user']['name']
        username = user['user']['username']
        total_books = user_rating_counts[user_id]

        f.write(f"{name} (@{username}) - ID: {user_id} - {total_books} books\n")

print(f"✓ Written {len(filtered_users)} users to {output_file}")
