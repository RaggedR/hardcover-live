#!/usr/bin/env python3
"""
Hardcover API - Invert User Books to Books with Users
Transforms user-centric data to book-centric data.
Input: user_books.json (users -> books)
Output: books_users.json (books -> users)
"""

import json
import os
from datetime import datetime, UTC
from collections import defaultdict


# File paths
INPUT_FILE = os.path.expanduser("~/data/hardcover/user_books.json")
OUTPUT_FILE = os.path.expanduser("~/data/hardcover/books_users.json")

# Status mapping
STATUS_NAMES = {
    1: "want_to_read",
    2: "currently_reading",
    3: "read",
    5: "did_not_finish"
}


def invert_data():
    """
    Invert the data structure from users->books to books->users.
    """
    print(f"Loading data from {INPUT_FILE}...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_books = data.get('user_books', [])
    print(f"✓ Loaded data for {len(user_books)} users\n")

    # Dictionary to store books with their users
    # Key: book_id, Value: book data with list of users
    books_dict = defaultdict(lambda: {
        "book": None,
        "users": []
    })

    total_books_processed = 0

    # Process each user's books
    for user_data in user_books:
        user_info = user_data.get('user', {})
        books = user_data.get('books')

        if not books:
            continue

        user_id = user_info.get('id')
        username = user_info.get('username')
        name = user_info.get('name')

        # Process all reading statuses
        for status_id, status_name in STATUS_NAMES.items():
            status_books = books.get(status_name, [])

            for book_entry in status_books:
                book = book_entry.get('book')
                if not book:
                    continue

                book_id = book.get('id')

                # Store book info if not already stored
                if books_dict[book_id]["book"] is None:
                    books_dict[book_id]["book"] = {
                        "id": book_id,
                        "title": book.get('title'),
                        "slug": book.get('slug'),
                        "image": book.get('image'),
                        "cached_contributors": book.get('cached_contributors')
                    }

                # Add user to this book's user list
                books_dict[book_id]["users"].append({
                    "user_id": user_id,
                    "username": username,
                    "name": name,
                    "status": status_name,
                    "status_id": status_id,
                    "rating": book_entry.get('rating'),
                    "review_raw": book_entry.get('review_raw')
                })

                total_books_processed += 1

    print(f"Processed {total_books_processed} book-user relationships")
    print(f"Found {len(books_dict)} unique books\n")

    # Convert to list format
    books_list = []
    for book_id, book_data in books_dict.items():
        books_list.append({
            "book": book_data["book"],
            "users": book_data["users"],
            "user_count": len(book_data["users"])
        })

    # Sort by number of users (most popular first)
    books_list.sort(key=lambda x: x["user_count"], reverse=True)

    # Create output data
    output_data = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "total_books": len(books_list),
            "total_relationships": total_books_processed,
            "source": "Inverted from user_books.json"
        },
        "books": books_list
    }

    # Save to file
    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Data saved successfully!\n")

    # Print summary
    print("Summary:")
    print(f"  Total unique books: {len(books_list)}")
    print(f"  Total book-user relationships: {total_books_processed}")

    if books_list:
        print(f"\nMost popular books:")
        for i, book in enumerate(books_list[:5], 1):
            title = book['book']['title']
            user_count = book['user_count']
            print(f"  {i}. '{title}' - {user_count} users")


def main():
    """Main execution function."""
    try:
        invert_data()
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found!", file=sys.stderr)
        print("Please run fetch_user_books.py first.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {INPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
