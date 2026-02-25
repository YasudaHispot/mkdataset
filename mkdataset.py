"""mkdataset - LLM CPT dataset builder from source code files.

Reads source code files from a directory, converts them to a HuggingFace
dataset in Parquet format suitable for continued pre-training (CPT).
Image files are converted to pixel array text representation.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from datasets import Dataset
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load TOML configuration file and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Approximate characters per token (fallback when no tokenizer is configured)
CHARS_PER_TOKEN = 3


def load_tokenizer(config: dict):
    """Load HuggingFace tokenizer if configured. Returns Tokenizer or None."""
    model_name = config.get("tokenizer")
    if not model_name:
        return None
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(model_name, "tokenizer.json")
    return Tokenizer.from_file(path)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text. Uses tokenizer if available, otherwise estimates."""
    if tokenizer is not None:
        return len(tokenizer.encode(text).ids)
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def is_binary_content(path: Path) -> bool:
    """Detect binary file by checking for null bytes in the first 8KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def read_file_safe(path: Path) -> str:
    """Read a text file with UTF-8, falling back to latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def read_image_as_text(path: Path) -> tuple[str, dict]:
    """Convert an image to pixel array text representation.

    Returns (text, image_info) where image_info contains width, height, channels.
    """
    img = Image.open(path).convert("RGBA")
    width, height = img.size
    pixels = list(img.tobytes())
    # Reconstruct as RGBA tuples from flat byte array
    pixel_tuples = [
        (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]) for i in range(0, len(pixels), 4)
    ]
    pixels = pixel_tuples
    rows = []
    for y in range(height):
        row = [list(pixels[y * width + x]) for x in range(width)]
        rows.append(row)
    text = str(rows)
    image_info = {"width": width, "height": height, "channels": "RGBA"}
    return text, image_info


def detect_language(path: Path, config: dict) -> str:
    """Detect language from file extension using config mappings."""
    ext = path.suffix.lower()
    if not ext:
        return "text"
    if ext in config.get("image_extensions", []):
        return "image-data"
    return config.get("language_map", {}).get(ext, "unknown")


def extract_project_name(rel_path: str) -> str:
    """Extract top-level directory name from relative path."""
    parts = Path(rel_path).parts
    if len(parts) > 1:
        return parts[0]
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_tokens: int, overlap_tokens: int, tokenizer=None) -> list[str]:
    """Split text into chunks by line boundaries respecting token limits.

    Returns a list of chunks. If the text fits in one chunk, returns [text].
    Splitting is done at line boundaries to avoid breaking code mid-line.
    Uses tokenizer for accurate token counting if available.
    """
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    lines = text.split("\n")
    chunks = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_with_nl = line + "\n"
        line_tokens = count_tokens(line_with_nl, tokenizer)
        if current_tokens + line_tokens > max_tokens and current_lines:
            chunks.append("\n".join(current_lines))
            # Overlap: keep trailing lines that fit within overlap_tokens
            overlap_lines: list[str] = []
            overlap_tok = 0
            for ol in reversed(current_lines):
                ol_tok = count_tokens(ol + "\n", tokenizer)
                if overlap_tok + ol_tok > overlap_tokens:
                    break
                overlap_lines.append(ol)
                overlap_tok += ol_tok
            overlap_lines.reverse()
            current_lines = overlap_lines
            current_tokens = overlap_tok
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def should_skip_file(path: Path, config: dict, max_size: int | None) -> bool:
    """Determine if a file should be skipped."""
    ext = path.suffix.lower()

    # Skip by extension
    if ext in config.get("skip_extensions", []):
        return True

    # Skip oversized files
    if max_size is not None:
        try:
            if path.stat().st_size > max_size:
                return True
        except OSError:
            return True

    # Image files are not skipped (they get converted)
    if ext in config.get("image_extensions", []):
        return False

    # Skip binary files
    if is_binary_content(path):
        return True

    return False


def collect_files(source_dir: Path, config: dict, max_size: int | None) -> list[Path]:
    """Collect all processable files from source directory."""
    source_dir = Path(source_dir)
    all_paths = sorted(p for p in source_dir.rglob("*") if p.is_file())
    files = []
    for path in tqdm(all_paths, desc="Scanning files"):
        if should_skip_file(path, config, max_size):
            continue

        ext = path.suffix.lower()
        is_image = ext in config.get("image_extensions", [])

        # Skip empty text files
        if not is_image:
            content = read_file_safe(path)
            if not content.strip():
                continue

        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_text_entry(
    content: str,
    rel_path: str,
    language: str,
    add_header: bool = True,
    image_info: dict | None = None,
) -> str:
    """Build the text column value with optional header."""
    if not add_header:
        return content

    lines = [f"### File: {rel_path}", f"### Language: {language}"]
    if image_info:
        lines.append(
            f"### Image: width={image_info['width']}, "
            f"height={image_info['height']}, "
            f"channels={image_info['channels']}"
        )
    lines.append("")
    lines.append(content)
    return "\n".join(lines)


def build_dataset(
    files: list[Path],
    source_dir: Path,
    config: dict,
    add_header: bool = True,
    tokenizer=None,
) -> Dataset:
    """Build a HuggingFace Dataset from collected files."""
    source_dir = Path(source_dir)
    max_tokens = config.get("max_tokens", 0)
    overlap_tokens = config.get("chunk_overlap_tokens", 256)

    records = {
        "text": [],
        "file_path": [],
        "language": [],
        "size_bytes": [],
        "num_lines": [],
        "num_tokens": [],
        "project": [],
    }

    def _append_record(text: str, rel_path: str, language: str, size_bytes: int, project: str):
        num_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        num_tokens = count_tokens(text, tokenizer)
        records["text"].append(text)
        records["file_path"].append(rel_path)
        records["language"].append(language)
        records["size_bytes"].append(size_bytes)
        records["num_lines"].append(num_lines)
        records["num_tokens"].append(num_tokens)
        records["project"].append(project)

    for path in tqdm(files, desc="Processing files"):
        rel_path = str(path.relative_to(source_dir))
        language = detect_language(path, config)
        size_bytes = path.stat().st_size
        project = extract_project_name(rel_path)

        if language == "image-data":
            content, image_info = read_image_as_text(path)
            text = build_text_entry(
                content=content,
                rel_path=rel_path,
                language=language,
                add_header=add_header,
                image_info=image_info,
            )
            _append_record(text, rel_path, language, size_bytes, project)
            continue

        content = read_file_safe(path)

        # Chunk splitting for large text files
        if max_tokens > 0:
            chunks = chunk_text(content, max_tokens, overlap_tokens, tokenizer)
        else:
            chunks = [content]

        for i, chunk in enumerate(chunks):
            chunk_path = f"{rel_path}#chunk-{i + 1}" if len(chunks) > 1 else rel_path
            text = build_text_entry(
                content=chunk,
                rel_path=chunk_path,
                language=language,
                add_header=add_header,
            )
            _append_record(text, chunk_path, language, size_bytes, project)

    return Dataset.from_dict(records)


def write_dataset(dataset: Dataset, output_dir: Path) -> None:
    """Write dataset to Parquet format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_dir / "train-00000-of-00001.parquet")


def print_token_stats(dataset: Dataset) -> None:
    """Print token statistics from dataset."""
    tokens = dataset["num_tokens"]
    paths = dataset["file_path"]
    max_idx = tokens.index(max(tokens))
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0
    print(f"\nMax tokens: {tokens[max_idx]:,} ({paths[max_idx]})")
    print(f"Avg tokens: {avg_tokens:,.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def dry_run(source_dir: Path, config: dict, max_size: int | None, tokenizer=None) -> None:
    """Show file listing and statistics without generating output."""
    files = collect_files(source_dir, config, max_size)

    lang_counts: dict[str, int] = {}
    project_counts: dict[str, int] = {}
    total_size = 0
    total_tokens = 0
    max_tokens_seen = 0
    max_tokens_file = ""

    for path in tqdm(files, desc="Analyzing files"):
        rel_path = str(path.relative_to(source_dir))
        language = detect_language(path, config)
        size = path.stat().st_size
        project = extract_project_name(rel_path)

        if language == "image-data":
            content, _ = read_image_as_text(path)
        else:
            content = read_file_safe(path)
        ntokens = count_tokens(content, tokenizer)

        lang_counts[language] = lang_counts.get(language, 0) + 1
        project_counts[project] = project_counts.get(project, 0) + 1
        total_size += size
        total_tokens += ntokens

        if ntokens > max_tokens_seen:
            max_tokens_seen = ntokens
            max_tokens_file = rel_path

        print(f"  {rel_path}  ({language}, {size:,} bytes, {ntokens:,} tokens)")

    print("\n--- Statistics ---")
    print(f"Total files: {len(files)}")
    print(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    avg_tokens = total_tokens / len(files) if files else 0
    print(f"\nMax tokens: {max_tokens_seen:,} ({max_tokens_file})")
    print(f"Avg tokens: {avg_tokens:,.0f}")
    tok_label = config.get("tokenizer", f"estimate (~{CHARS_PER_TOKEN} chars/token)")
    print(f"Tokenizer: {tok_label}")
    print("\nBy language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
    print("\nBy project:")
    for proj, count in sorted(project_counts.items(), key=lambda x: -x[1]):
        print(f"  {proj}: {count}")


def resolve_source_dir(cli_source: str | None, config: dict) -> Path:
    """Resolve source directory from CLI arg and config.

    Priority: CLI --source > config source_dir > current directory "."
    """
    if cli_source is not None:
        return Path(cli_source)
    return Path(config.get("source_dir", "."))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build LLM CPT dataset from source code files.")
    parser.add_argument(
        "-s",
        "--source",
        default=None,
        help="Source directory (default: config source_dir or current directory)",
    )
    parser.add_argument(
        "-o", "--output", default="./output", help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "-c", "--config", default="./config.toml", help="Config file path (default: ./config.toml)"
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
        help="Max file size in bytes (default: unlimited)",
    )
    parser.add_argument("--no-header", action="store_true", help="Don't add headers to text column")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show file listing without generating output"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)

    source_dir = resolve_source_dir(args.source, config)
    output_dir = Path(args.output)

    if not source_dir.is_dir():
        print(f"Error: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    tokenizer = load_tokenizer(config)
    tok_label = config.get("tokenizer", f"estimate (~{CHARS_PER_TOKEN} chars/token)")
    print(f"Tokenizer: {tok_label}")

    if args.dry_run:
        dry_run(source_dir, config, args.max_file_size, tokenizer)
        return

    print(f"Source: {source_dir.resolve()}")
    files = collect_files(source_dir, config, args.max_file_size)
    print(f"Found {len(files)} files to process.")

    if not files:
        print("No files to process. Exiting.")
        return

    add_header = not args.no_header
    ds = build_dataset(files, source_dir, config, add_header=add_header, tokenizer=tokenizer)

    print(f"Writing dataset to {output_dir}...")
    write_dataset(ds, output_dir)

    print(f"\nDone! Dataset has {len(ds)} records.")
    print(f"  Columns: {ds.column_names}")
    languages = {}
    for lang in ds["language"]:
        languages[lang] = languages.get(lang, 0) + 1
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
    print_token_stats(ds)


if __name__ == "__main__":
    main()
