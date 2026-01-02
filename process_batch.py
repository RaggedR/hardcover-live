#!/usr/bin/env python3
"""
Hardcover API - Batch Processor
Coordinates all 4 phases of processing a batch of 25 users:
1. Download 25 users
2. Get books for those 25 users
3. Create inverted JSON (books -> users)
4. Print statistics
Only updates progress after ALL 4 phases complete.
"""

import json
import os
import sys
import time
from datetime import datetime, UTC
from collections import defaultdict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_ENDPOINT = "https://api.hardcover.app/v1/graphql"
API_TOKEN = os.getenv("HARDCOVER_API_TOKEN", "YOUR_API_TOKEN_HERE")
USERS_FILE = os.path.expanduser("~/data/hardcover/users.json")
USER_BOOKS_FILE = os.path.expanduser("~/data/hardcover/user_books.json")
BOOKS_USERS_FILE = os.path.expanduser("~/data/hardcover/books_users.json")
PROGRESS_FILE = os.path.expanduser("~/git/hardcover/progress.json")
BATCH_SIZE = 25
REQUEST_DELAY = 1.0


def load_progress():
    """Load progress file."""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "batches_processed": 0,
            "total_users": 0,
            "total_books": 0,
            "last_updated": None
        }


def save_progress(progress):
    """Save progress file."""
    progress["last_updated"] = datetime.now(UTC).isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def load_json_file(filepath, default):
    """Load JSON file or return default."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json_file(filepath, data):
    """Save JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# PHASE 1: Download 25 users
# ============================================================================

def fetch_users(offset):
    """Fetch 25 users from API."""
    query = """
    query GetUsers($limit: Int!, $offset: Int!) {
      users(limit: $limit, offset: $offset, order_by: {created_at: desc}) {
        id
        name
        username
        bio
        image { url }
        created_at
        updated_at
      }
    }
    """

    payload = {
        "query": query,
        "variables": {"limit": BATCH_SIZE, "offset": offset}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")

    return data.get("data", {}).get("users", [])


# ============================================================================
# PHASE 2: Get books for each user
# ============================================================================

def fetch_user_books(user_id):
    """Fetch books for a specific user."""
    query = """
    query GetUserBooks($user_id: Int!) {
      user_books(where: {user_id: {_eq: $user_id}}) {
        id
        status_id
        rating
        review_raw
        book {
          id
          title
          slug
          image { url }
          cached_contributors
        }
      }
    }
    """

    payload = {
        "query": query,
        "variables": {"user_id": user_id}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        return []

    user_books = data.get("data", {}).get("user_books", [])

    # Organize by status
    result = {
        "want_to_read": [],
        "currently_reading": [],
        "read": [],
        "other": []
    }

    for entry in user_books:
        status_id = entry.get("status_id")
        if status_id == 1:
            result["want_to_read"].append(entry)
        elif status_id == 2:
            result["currently_reading"].append(entry)
        elif status_id == 3:
            result["read"].append(entry)
        else:
            result["other"].append(entry)

    return result


# ============================================================================
# PHASE 3: Update inverted JSON (books -> users) - INCREMENTAL
# ============================================================================

def update_books_users_json(existing_books_list, new_user_books_data):
    """
    Update inverted structure with NEW users only (incremental).

    Args:
        existing_books_list: Existing books from books_users.json
        new_user_books_data: Only the NEW 25 users from this batch

    Returns:
        Updated books list
    """
    # Convert existing list to dict for efficient merging
    books_dict = {}
    for book_entry in existing_books_list:
        book_id = book_entry["book"]["id"]
        books_dict[book_id] = {
            "book": book_entry["book"],
            "users": book_entry["users"]
        }

    STATUS_NAMES = {
        1: "want_to_read",
        2: "currently_reading",
        3: "read",
        5: "did_not_finish"
    }

    # Process ONLY the new users
    for user_data in new_user_books_data:
        user_info = user_data["user"]
        books = user_data.get("books")

        if not books:
            continue

        for status_id, status_name in STATUS_NAMES.items():
            status_books = books.get(status_name, [])

            for book_entry in status_books:
                book = book_entry.get("book")
                if not book:
                    continue

                book_id = book["id"]

                # Create book entry if it doesn't exist
                if book_id not in books_dict:
                    books_dict[book_id] = {
                        "book": {
                            "id": book_id,
                            "title": book.get("title"),
                            "slug": book.get("slug"),
                            "image": book.get("image"),
                            "cached_contributors": book.get("cached_contributors")
                        },
                        "users": []
                    }

                # Add this user to the book
                books_dict[book_id]["users"].append({
                    "user_id": user_info["id"],
                    "username": user_info["username"],
                    "name": user_info["name"],
                    "status": status_name,
                    "status_id": status_id,
                    "rating": book_entry.get("rating"),
                    "review_raw": book_entry.get("review_raw")
                })

    # Convert to list and sort by popularity
    books_list = []
    for book_id, book_data in books_dict.items():
        books_list.append({
            "book": book_data["book"],
            "users": book_data["users"],
            "user_count": len(book_data["users"])
        })

    books_list.sort(key=lambda x: x["user_count"], reverse=True)

    return books_list


# ============================================================================
# PHASE 4: Calculate and print statistics
# ============================================================================

def print_statistics(books_list, user_books_data):
    """Print required statistics."""
    # a) Percentage of books with more than one user
    total_books = len(books_list)
    books_with_multiple_users = sum(1 for b in books_list if b["user_count"] > 1)

    if total_books > 0:
        percentage = (books_with_multiple_users / total_books) * 100
    else:
        percentage = 0

    # b) Average number of books per user
    total_users = len(user_books_data)
    total_book_entries = sum(
        len(u.get("books", {}).get("read", [])) +
        len(u.get("books", {}).get("currently_reading", [])) +
        len(u.get("books", {}).get("want_to_read", [])) +
        len(u.get("books", {}).get("other", []))
        for u in user_books_data
    )

    if total_users > 0:
        avg_books = total_book_entries / total_users
    else:
        avg_books = 0

    print(f"\n{'='*60}")
    print(f"STATISTICS")
    print(f"{'='*60}")
    print(f"a) Books with >1 user: {percentage:.1f}% ({books_with_multiple_users}/{total_books})")
    print(f"b) Average books per user: {avg_books:.1f}")
    print(f"c) Number of users processed: {total_users}")
    print(f"d) Number of books: {total_books}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN COORDINATOR
# ============================================================================

def main():
    """Main coordinator for all 4 phases."""

    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("ERROR: Set HARDCOVER_API_TOKEN in .env file!")
        sys.exit(1)

    # Load progress
    progress = load_progress()
    batch_num = progress["batches_processed"] + 1
    offset = progress["batches_processed"] * BATCH_SIZE

    print(f"\n{'='*60}")
    print(f"PROCESSING BATCH #{batch_num}")
    print(f"{'='*60}")
    print(f"Batches already processed: {progress['batches_processed']}")
    print(f"Total users so far: {progress['total_users']}")
    print(f"{'='*60}\n")

    # ========================================================================
    # PHASE 1: Download 25 users
    # ========================================================================
    print(f"PHASE 1: Downloading 25 users (offset {offset})...")
    try:
        new_users = fetch_users(offset)
        if not new_users:
            print("No more users available!")
            sys.exit(0)
        print(f"✓ Downloaded {len(new_users)} users\n")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)

    # Load and update users.json
    users_data = load_json_file(USERS_FILE, {"metadata": {}, "users": []})
    users_data["users"].extend(new_users)
    users_data["metadata"]["count"] = len(users_data["users"])
    save_json_file(USERS_FILE, users_data)

    # ========================================================================
    # PHASE 2: Get books for each user
    # ========================================================================
    print(f"PHASE 2: Fetching books for {len(new_users)} users...")
    new_user_books = []

    for idx, user in enumerate(new_users, 1):
        print(f"  [{idx}/{len(new_users)}] {user['name']} (@{user['username']})...", end=" ")

        books = fetch_user_books(user['id'])
        counts = {
            "read": len(books.get("read", [])),
            "currently_reading": len(books.get("currently_reading", [])),
            "want_to_read": len(books.get("want_to_read", [])),
            "other": len(books.get("other", []))
        }
        total = sum(counts.values())

        print(f"{total} books")

        new_user_books.append({
            "user": {
                "id": user["id"],
                "name": user["name"],
                "username": user["username"]
            },
            "books": books,
            "counts": counts
        })

        if idx < len(new_users):
            time.sleep(REQUEST_DELAY)

    print(f"✓ Fetched books for all users\n")

    # Load and update user_books.json
    user_books_data = load_json_file(USER_BOOKS_FILE, {"metadata": {}, "user_books": []})
    user_books_data["user_books"].extend(new_user_books)
    user_books_data["metadata"]["total_users"] = len(user_books_data["user_books"])
    save_json_file(USER_BOOKS_FILE, user_books_data)

    # ========================================================================
    # PHASE 3: Update inverted JSON (incremental)
    # ========================================================================
    print(f"PHASE 3: Updating books->users JSON (incremental)...")

    # Load existing books_users.json
    existing_books_data = load_json_file(BOOKS_USERS_FILE, {"metadata": {}, "books": []})
    existing_books_list = existing_books_data.get("books", [])

    # Update with ONLY the new 25 users
    books_list = update_books_users_json(existing_books_list, new_user_books)

    books_users_data = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "total_books": len(books_list),
            "total_users": len(user_books_data["user_books"])
        },
        "books": books_list
    }

    save_json_file(BOOKS_USERS_FILE, books_users_data)
    print(f"✓ Updated books_users.json with {len(books_list)} books\n")

    # ========================================================================
    # PHASE 4: Print statistics
    # ========================================================================
    print(f"PHASE 4: Calculating statistics...")
    print_statistics(books_list, user_books_data["user_books"])

    # ========================================================================
    # UPDATE PROGRESS - Only after all 4 phases complete
    # ========================================================================
    progress["batches_processed"] = batch_num
    progress["total_users"] = len(users_data["users"])
    progress["total_books"] = len(books_list)
    save_progress(progress)

    print(f"✓ Progress updated: Batch #{batch_num} complete")
    print(f"\nRun this script again to process the next batch.\n")


if __name__ == "__main__":
    main()
