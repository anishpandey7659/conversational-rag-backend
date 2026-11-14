from app.schemas.document import ChunkingStrategy


def _validate_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")


def fixed_size_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into fixed-width pieces with the requested character overlap."""
    _validate_options(chunk_size, chunk_overlap)
    source = text.strip()
    if not source:
        return []
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(source):
        chunks.append(source[start : start + chunk_size])
        if start + chunk_size >= len(source):
            break
        start += step
    return chunks


def recursive_character_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Prefer paragraph, line, and word boundaries before hard character cuts."""
    _validate_options(chunk_size, chunk_overlap)
    source = text.strip()
    if not source:
        return []

    separators = ("\n\n", "\n", " ", "")

    def find_boundary(start: int, end: int, remaining: tuple[str, ...]) -> int:
        if not remaining:
            return end
        separator = remaining[0]
        if separator:
            boundary = source.rfind(separator, start + 1, end)
            if boundary > start:
                return boundary + len(separator)
        return find_boundary(start, end, remaining[1:])

    chunks: list[str] = []
    start = 0

    while start < len(source):
        end = min(start + chunk_size, len(source))
        if end < len(source):
            end = find_boundary(start, end, separators)

        chunk = source[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(source):
            break

        next_start = max(start + 1, end - chunk_overlap)
        start = next_start

    return chunks


def split_text(
    text: str,
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if strategy == ChunkingStrategy.fixed:
        return fixed_size_chunks(text, chunk_size, chunk_overlap)
    if strategy == ChunkingStrategy.recursive:
        return recursive_character_chunks(text, chunk_size, chunk_overlap)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
