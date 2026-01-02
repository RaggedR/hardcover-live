#!/usr/bin/env python3
"""
Hardcover API - Append More Users
Fetches the next batch of users and appends them to the existing users.json file.
"""

import json
import os
import sys
from datetime import datetime, UTC
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_ENDPOINT = "https://api.hardcover.app/v1/graphql"
API_TOKEN = os.getenv("HARDCOVER_API_TOKEN", "YOUR_API_TOKEN_HERE")
USERS_FILE = os.path.expanduser("~/data/hardcover/users.json")


def load_existing_users():
    """Load existing users from the users.json file."""
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: {USERS_FILE} not found!", file=sys.stderr)
        sys.exit(1)


def fetch_users(limit=25, offset=0):
    """
    Fetch users from Hardcover API with offset.

    Args:
        limit: Number of users to fetch (default: 25)
        offset: Number of users to skip (default: 0)

    Returns:
        dict: API response data
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

    print(f"Fetching {limit} users (offset: {offset})...")

    try:
        response = requests.post(API_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()

        if "errors" in data:
            print(f"GraphQL errors: {data['errors']}", file=sys.stderr)
            return None

        return data

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return None


def main():
    """Main execution function."""

    # Check if API token is configured
    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("ERROR: Please set your Hardcover API token in .env file!", file=sys.stderr)
        sys.exit(1)

    # Load existing users
    print(f"Loading existing users from {USERS_FILE}...")
    existing_data = load_existing_users()
    existing_users = existing_data.get('users', [])
    current_count = len(existing_users)
    print(f"✓ Found {current_count} existing users\n")

    # Fetch next batch using offset
    result = fetch_users(limit=25, offset=current_count)

    if result is None:
        print("Failed to fetch new users", file=sys.stderr)
        sys.exit(1)

    new_users = result.get("data", {}).get("users", [])

    if not new_users:
        print("No new users found!")
        sys.exit(0)

    print(f"✓ Fetched {len(new_users)} new users\n")

    # Show new users
    print("New users:")
    for user in new_users:
        print(f"  - {user['name']} (@{user['username']}) - ID: {user['id']}")

    # Append new users to existing list
    existing_users.extend(new_users)

    # Update metadata
    existing_data['metadata']['count'] = len(existing_users)
    existing_data['metadata']['last_updated'] = datetime.now(UTC).isoformat()
    existing_data['users'] = existing_users

    # Save updated data
    print(f"\nSaving updated data to {USERS_FILE}...")
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Data saved successfully!")
    print(f"\nTotal users now: {len(existing_users)} ({current_count} + {len(new_users)})")


if __name__ == "__main__":
    main()
