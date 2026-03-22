import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, AGENT_VECTOR_CHUNK_SIZE: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split text recursively by ["\\n\\n", "\\n", " ", ""] to ensure chunks 
    do not exceed AGENT_VECTOR_CHUNK_SIZE, while maintaining a chunk_overlap of characters.
    Mimics LangChain's RecursiveCharacterTextSplitter natively.
    """
    if not text:
        return []
        
    separators = ["\n\n", "\n", " ", ""]
    return _split_recursively(text, separators, AGENT_VECTOR_CHUNK_SIZE, chunk_overlap)


def _split_recursively(text: str, separators: list[str], AGENT_VECTOR_CHUNK_SIZE: int, chunk_overlap: int) -> list[str]:
    if len(text) <= AGENT_VECTOR_CHUNK_SIZE:
        return [text]

    # Find the appropriate separator for this recursion depth
    separator = separators[-1]
    next_separators = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            next_separators = separators[i+1:] if sep != "" else []
            break

    # Split the text
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    # Recursively break down any splits that are still too large
    good_splits = []
    for s in splits:
        if len(s) <= AGENT_VECTOR_CHUNK_SIZE:
            good_splits.append(s)
        else:
            if next_separators:
                good_splits.extend(_split_recursively(s, next_separators, AGENT_VECTOR_CHUNK_SIZE, chunk_overlap))
            else:
                for j in range(0, len(s), AGENT_VECTOR_CHUNK_SIZE):
                    good_splits.append(s[j:j+AGENT_VECTOR_CHUNK_SIZE])

    # Merge splits into chunks respecting AGENT_VECTOR_CHUNK_SIZE and chunk_overlap
    chunks = []
    current_chunk = []
    current_length = 0
    separator_len = len(separator)

    for s in good_splits:
        s_len = len(s)
        # Check if adding this split exceeds AGENT_VECTOR_CHUNK_SIZE
        if current_length + s_len + (separator_len if current_chunk else 0) > AGENT_VECTOR_CHUNK_SIZE and current_chunk:
            chunks.append(separator.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_chunk = []
            overlap_length = 0
            for item in reversed(current_chunk):
                item_len = len(item) + (separator_len if overlap_chunk else 0)
                if overlap_length + item_len > chunk_overlap:
                    break
                overlap_chunk.insert(0, item)
                overlap_length += item_len
                
            current_chunk = overlap_chunk
            current_length = overlap_length
        
        current_chunk.append(s)
        current_length += s_len + (separator_len if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks
