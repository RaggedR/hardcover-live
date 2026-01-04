#!/usr/bin/env python3
"""
Hardcover API - Fetch User Books with Progress Tracking
Automatically tracks which users have been processed.
Uses book_progress.json to know where to continue from.
"""

import json
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_ENDPOINT = "https://api.hardcover.app/v1/graphql"
API_TOKEN = os.getenv("HARDCOVER_API_TOKEN", "YOUR_API_TOKEN_HERE")
USERS_FILE = os.path.expanduser("~/data/hardcover/users.json")
USER_BOOKS_FILE = os.path.expanduser("~/data/hardcover/user_books.json")
BOOK_PROGRESS_FILE = os.path.expanduser("~/git/hardcover/book_progress.json")
REQUEST_DELAY = 1.0  # Delay between requests to avoid rate limiting


def load_book_progress():
    """Load book fetching progress."""
    try:
        with open(BOOK_PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # First run - initialize progress
        return {
            "users_processed": 0,
            "total_books_fetched": 0,
            "last_updated": None
        }


def save_book_progress(progress):
    """Save book fetching progress."""
    progress["last_updated"] = datetime.now(UTC).isoformat()
    with open(BOOK_PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_users():
    """Load users from the users.json file."""
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['users']


def load_existing_user_books():
    """Load existing user books data."""
    try:
        with open(USER_BOOKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "metadata": {
                "fetched_at": datetime.now(UTC).isoformat(),
                "total_users": 0,
                "successful_fetches": 0,
                "source": "Hardcover GraphQL API"
            },
            "user_books": []
        }


def fetch_user_books(user_id, username):
    """
    Fetch all books for a specific user by reading status.

    Args:
        user_id: User's ID
        username: User's username for logging

    Returns:
        dict: Books organized by status (read, currently_reading, want_to_read)
    """
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
          image {
            url
          }
          cached_contributors
        }
      }
    }
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    payload = {
        "query": query,
        "variables": {"user_id": user_id}
    }

    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        if "errors" in data:
            print(f"  ⚠ GraphQL errors: {data['errors']}", file=sys.stderr)
            return None

        # Organize books by status
        user_books = data.get("data", {}).get("user_books", [])

        result = {
            "want_to_read": [],
            "currently_reading": [],
            "read": [],
            "other": []
        }

        for book_entry in user_books:
            status_id = book_entry.get("status_id")
            if status_id == 1:
                result["want_to_read"].append(book_entry)
            elif status_id == 2:
                result["currently_reading"].append(book_entry)
            elif status_id == 3:
                result["read"].append(book_entry)
            else:
                result["other"].append(book_entry)

        counts = {
            "want_to_read": len(result["want_to_read"]),
            "currently_reading": len(result["currently_reading"]),
            "read": len(result["read"]),
            "other": len(result["other"]),
            "total": len(user_books)
        }

        return {
            "books": result,
            "counts": counts
        }

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request failed: {e}", file=sys.stderr)
        return None


def main():
    """Main execution function."""

    # Check if API token is configured
    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("ERROR: Please set your Hardcover API token in .env file!", file=sys.stderr)
        sys.exit(1)

    # Load progress
    print("Checking book fetching progress...")
    progress = load_book_progress()
    users_processed = progress["users_processed"]

    print(f"✓ Progress: {users_processed} users already processed")
    print(f"  Total books fetched so far: {progress['total_books_fetched']}\n")

    # Load all users
    print(f"Loading users from {USERS_FILE}...")
    try:
        all_users = load_users()
        total_users = len(all_users)
        print(f"✓ Found {total_users} total users in file\n")
    except FileNotFoundError:
        print(f"ERROR: {USERS_FILE} not found!", file=sys.stderr)
        print("Please run fetch_users.py first.", file=sys.stderr)
        sys.exit(1)

    # Determine which users to process (next batch of 25)
    BATCH_SIZE = 25
    users_to_process = all_users[users_processed:users_processed + BATCH_SIZE]

    if not users_to_process:
        print("All users have been processed!")
        print("Run fetch_users.py to get more users.")
        sys.exit(0)

    end_index = min(users_processed + len(users_to_process), total_users)
    print(f"Processing {len(users_to_process)} users (#{users_processed + 1} to #{end_index})...\n")

    # Load existing user books data
    existing_data = load_existing_user_books()
    existing_user_books = existing_data.get("user_books", [])

    # Fetch books for new users
    new_results = []
    total_new_books = 0
    successful_fetches = 0

    for idx, user in enumerate(users_to_process, 1):
        user_id = user['id']
        username = user['username']
        name = user['name']

        print(f"[{idx}/{len(users_to_process)}] {name} (@{username})...", end=" ")

        user_books_data = fetch_user_books(user_id, username)

        if user_books_data:
            counts = user_books_data["counts"]
            print(f"✓ {counts['total']} books ({counts['read']} read, "
                  f"{counts['currently_reading']} reading, {counts['want_to_read']} want)")

            new_results.append({
                "user": {
                    "id": user_id,
                    "name": name,
                    "username": username
                },
                "books": user_books_data["books"],
                "counts": user_books_data["counts"]
            })

            total_new_books += counts['total']
            successful_fetches += 1
        else:
            print(f"✗ Failed")
            new_results.append({
                "user": {
                    "id": user_id,
                    "name": name,
                    "username": username
                },
                "books": None,
                "counts": None,
                "error": "Failed to fetch books"
            })

        # Rate limiting
        if idx < len(users_to_process):
            time.sleep(REQUEST_DELAY)

    # Append new results to existing data
    existing_user_books.extend(new_results)

    # Update metadata
    existing_data["metadata"]["total_users"] = len(existing_user_books)
    existing_data["metadata"]["successful_fetches"] = sum(
        1 for r in existing_user_books if r.get("books") is not None
    )
    existing_data["metadata"]["last_updated"] = datetime.now(UTC).isoformat()
    existing_data["user_books"] = existing_user_books

    # Save updated user books file
    print(f"\nSaving to {USER_BOOKS_FILE}...")
    with open(USER_BOOKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Data saved successfully!")

    # Update progress
    progress["users_processed"] = users_processed + len(users_to_process)
    progress["total_books_fetched"] = progress["total_books_fetched"] + total_new_books
    save_book_progress(progress)
    print(f"✓ Progress updated: {progress['users_processed']} users processed")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed {len(users_to_process)} new users")
    print(f"  Fetched {total_new_books} new books")
    print(f"  Total users processed: {progress['users_processed']}/{total_users}")
    print(f"  Total books fetched: {progress['total_books_fetched']}")
    if progress['users_processed'] < total_users:
        print(f"\n  Run this script again to process the next batch")
    else:
        print(f"\n  All users processed! Run fetch_users.py to get more users")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
