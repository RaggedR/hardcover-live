#!/usr/bin/env python3
"""
Hardcover API - Fetch Users with Progress Tracking
Automatically tracks progress and fetches the next batch of users.
Uses progress.json to know where to continue from.
"""

import json
import os
import sys
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
PROGRESS_FILE = os.path.expanduser("~/git/hardcover/progress.json")
BATCH_SIZE = 25


def load_progress():
    """Load progress from progress.json file."""
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # First run - initialize progress
        return {
            "batches_downloaded": 0,
            "users_per_batch": BATCH_SIZE,
            "total_users": 0,
            "last_updated": None
        }


def save_progress(progress):
    """Save progress to progress.json file."""
    progress["last_updated"] = datetime.now(UTC).isoformat()
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_existing_users():
    """Load existing users from users.json file."""
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "metadata": {
                "fetched_at": datetime.now(UTC).isoformat(),
                "count": 0,
                "source": "Hardcover GraphQL API"
            },
            "users": []
        }


def fetch_users(limit=25, offset=0):
    """
    Fetch users from Hardcover API.

    Args:
        limit: Number of users to fetch (default: 25)
        offset: Number of users to skip (default: 0)

    Returns:
        list: User data
    """
    query = """
    query GetUsers($limit: Int!, $offset: Int!) {
      users(limit: $limit, offset: $offset, order_by: {created_at: desc}) {
        id
        name
        username
        bio
        image {
          url
        }
        created_at
        updated_at
      }
    }
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    payload = {
        "query": query,
        "variables": {
            "limit": limit,
            "offset": offset
        }
    }

    print(f"Fetching {limit} users (offset: {offset}, batch #{offset // limit + 1})...")

    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        if "errors" in data:
            print(f"GraphQL errors: {data['errors']}", file=sys.stderr)
            return None

        return data.get("data", {}).get("users", [])

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return None


def main():
    """Main execution function."""

    # Check if API token is configured
    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("ERROR: Please set your Hardcover API token in .env file!", file=sys.stderr)
        sys.exit(1)

    # Load progress
    print("Checking progress...")
    progress = load_progress()
    batches_done = progress["batches_downloaded"]
    offset = batches_done * BATCH_SIZE

    print(f"✓ Progress: {batches_done} batches downloaded ({progress['total_users']} users)")
    print(f"  Next batch will start at offset {offset}\n")

    # Load existing users
    existing_data = load_existing_users()
    existing_users = existing_data.get("users", [])

    # Fetch next batch
    new_users = fetch_users(limit=BATCH_SIZE, offset=offset)

    if new_users is None:
        print("Failed to fetch users", file=sys.stderr)
        sys.exit(1)

    if not new_users:
        print("No more users to fetch!")
        sys.exit(0)

    print(f"✓ Fetched {len(new_users)} new users\n")

    # Show new users
    print("New users:")
    for user in new_users[:10]:  # Show first 10
        print(f"  - {user['name']} (@{user['username']}) - ID: {user['id']}")
    if len(new_users) > 10:
        print(f"  ... and {len(new_users) - 10} more")

    # Append new users
    existing_users.extend(new_users)

    # Update metadata
    existing_data["metadata"]["count"] = len(existing_users)
    existing_data["metadata"]["last_updated"] = datetime.now(UTC).isoformat()
    existing_data["users"] = existing_users

    # Save updated users file
    print(f"\nSaving to {USERS_FILE}...")
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Data saved successfully!")

    # Update progress
    progress["batches_downloaded"] += 1
    progress["total_users"] = len(existing_users)
    save_progress(progress)
    print(f"✓ Progress updated: {progress['batches_downloaded']} batches, {progress['total_users']} total users")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Batch #{progress['batches_downloaded']} completed")
    print(f"  Total users: {progress['total_users']}")
    print(f"  Run this script again to fetch the next batch of 25 users")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
