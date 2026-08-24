"""Prefix artist tags in caption files using a compact artist-only database."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_ARTIST_DATABASE = Path(__file__).resolve().with_name("artist_tags.db")


def normalize_tag(tag: object) -> str:
    """Normalize underscores, escaped parentheses, whitespace, and case."""
    if tag is None:
        return ""

    value = str(tag).strip()
    value = value.replace(r"\(", "(").replace(r"\)", ")")
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def display_tag(tag: object) -> str:
    """Clean a tag while preserving readable capitalization."""
    value = str(tag).strip()
    value = value.replace(r"\(", "(").replace(r"\)", ")")
    value = value.replace("_", " ")
    return re.sub(r"\s+", " ", value)


def load_artist_tags(database_path: Path) -> dict[str, str]:
    """Load the compact ``artist_tags`` database into memory."""
    database_path = database_path.expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Artist database not found: {database_path}")

    uri = f"{database_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(artist_tags)")
        }
        required_columns = {"normalized_tag", "display_tag"}
        if not required_columns.issubset(columns):
            raise ValueError(
                "The database is not a compatible artist-only database. "
                "Create it with create_artist_tag_database.py."
            )

        artist_tags = dict(
            connection.execute(
                "SELECT normalized_tag, display_tag FROM artist_tags"
            )
        )
    finally:
        connection.close()

    if not artist_tags:
        raise ValueError("The artist database contains no artist tags.")

    print(f"Loaded {len(artist_tags):,} unique artist tags from {database_path}")
    return artist_tags


def iter_caption_files(dataset_dir: Path) -> Iterator[Path]:
    """Recursively yield .txt caption files in deterministic order."""
    for root, directory_names, filenames in os.walk(dataset_dir):
        directory_names.sort(key=str.casefold)
        for filename in sorted(filenames, key=str.casefold):
            if filename.lower().endswith(".txt"):
                yield Path(root) / filename


def parse_caption_tags(content: str) -> list[str]:
    """Split a booru-style caption on commas and line endings."""
    content = content.replace(r"\(", "(").replace(r"\)", ")")
    return [
        tag.strip()
        for tag in re.split(r"[,\r\n]+", content)
        if tag.strip()
    ]


def deduplicate_tags(tags: Iterable[str]) -> list[str]:
    """Deduplicate tags case-insensitively while preserving their order."""
    result: list[str] = []
    seen: set[str] = set()

    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(display_tag(tag))

    return result


def write_text_atomically(path: Path, content: str) -> None:
    """Replace a caption atomically so interruption cannot leave a partial file."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def remove_artist_prefix(tag: str) -> str | None:
    """Return the value following a leading @, or None if there is no prefix."""
    match = re.match(r"^@\s*(.+)$", display_tag(tag))
    return match.group(1).strip() if match else None


def process_caption(
    file_path: Path,
    artist_tags: dict[str, str],
    dry_run: bool = False,
) -> tuple[bool, int]:
    """Format artist tags in one caption and return (changed, artist count)."""
    original_content = file_path.read_text(encoding="utf-8-sig")
    raw_tags = parse_caption_tags(original_content)
    if not raw_tags:
        return False, 0

    ordinary_tags: list[str] = []
    detected_artists: list[str] = []
    detected_artist_keys: set[str] = set()

    for original_tag in raw_tags:
        prefixed_artist = remove_artist_prefix(original_tag)
        if prefixed_artist is not None:
            artist_key = normalize_tag(prefixed_artist)
            if artist_key and artist_key not in detected_artist_keys:
                detected_artist_keys.add(artist_key)
                detected_artists.append(
                    artist_tags.get(artist_key, display_tag(prefixed_artist))
                )
            continue

        normalized_tag = normalize_tag(original_tag)
        if not normalized_tag:
            continue

        if normalized_tag in artist_tags:
            if normalized_tag not in detected_artist_keys:
                detected_artist_keys.add(normalized_tag)
                detected_artists.append(artist_tags[normalized_tag])
        else:
            ordinary_tags.append(display_tag(original_tag))

    final_tags = deduplicate_tags(ordinary_tags)
    present_normalized = {normalize_tag(tag) for tag in final_tags}

    for artist in detected_artists:
        formatted_artist = f"@ {artist}"
        normalized_formatted = normalize_tag(formatted_artist)
        if normalized_formatted not in present_normalized:
            final_tags.append(formatted_artist)
            present_normalized.add(normalized_formatted)

    final_content = ", ".join(final_tags)
    changed = final_content != original_content.strip()

    if changed and not dry_run:
        write_text_atomically(file_path, final_content)

    return changed, len(detected_artists)


def add_artist_prefix_to_captions(
    dataset_dir: Path,
    database_path: Path,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    """Process a caption tree and return checked, changed, artists, failures."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")

    artist_tags = load_artist_tags(database_path)
    checked_count = 0
    changed_count = 0
    failed_count = 0
    artist_count = 0

    print("\n--- Starting Caption Artist Formatting ---")
    if dry_run:
        print("Dry run: captions will not be modified.")

    for file_path in iter_caption_files(dataset_dir):
        try:
            changed, file_artist_count = process_caption(
                file_path=file_path,
                artist_tags=artist_tags,
                dry_run=dry_run,
            )
            checked_count += 1
            changed_count += int(changed)
            artist_count += file_artist_count

            if checked_count % 500 == 0:
                print(
                    f"Checked {checked_count:,} files...",
                    end="\r",
                    flush=True,
                )
        except (OSError, UnicodeError) as exc:
            failed_count += 1
            print(f"\n[WARNING] Could not process '{file_path}': {exc}")

    action = "would be updated" if dry_run else "updated"
    print(" " * 80, end="\r")
    print("--- Processing Complete ---")
    print(f"Caption files checked:       {checked_count:,}")
    print(f"Caption files {action}: {changed_count:,}")
    print(f"Artist occurrences detected: {artist_count:,}")
    print(f"Files that failed:           {failed_count:,}")
    return checked_count, changed_count, artist_count, failed_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add an '@ ' prefix to artist tags in caption files using the "
            "compact artist-only database."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        help="Directory containing .txt caption files.",
    )
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=DEFAULT_ARTIST_DATABASE,
        help=f"Artist-only SQLite database (default: {DEFAULT_ARTIST_DATABASE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying caption files.",
    )
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    dataset_dir = arguments.dataset_dir

    if dataset_dir is None:
        print("Dataset Caption Artist Prefix Tool")
        entered_path = input("Enter the path to your dataset directory: ")
        dataset_dir = Path(entered_path.strip().strip('"'))

    try:
        _, _, _, failed_count = add_artist_prefix_to_captions(
            dataset_dir=dataset_dir,
            database_path=arguments.database,
            dry_run=arguments.dry_run,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
