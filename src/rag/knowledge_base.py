import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class MedicalChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class MedicalKnowledgeBaseBuilder:
    def __init__(self):
        self.chunks: List[MedicalChunk] = []

    def _parse_front_matter(self, content: str) -> tuple[Dict[str, Any], str]:
        """
        Extracts YAML-style front matter between --- blocks.
        Returns a dictionary of metadata and the remaining markdown text content.

        Note: this is a flat line-by-line parser — it does NOT support nested
        structures or list values (e.g. `tags: [skin, bacterial]`). Keep frontmatter
        schema flat (string/int/float scalars only) or this will mis-parse.
        """
        metadata = {}
        markdown_content = content

        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if match:
            front_matter = match.group(1)
            markdown_content = content[match.end():]

            for line in front_matter.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()

                    if val.isdigit():
                        val = int(val)
                    elif val.replace('.', '', 1).isdigit() and val.count('.') == 1:
                        val = float(val)

                    metadata[key] = val

        return metadata, markdown_content

    def _make_chunk_id(self, file_stem: str, section_header: str) -> str:
        """
        Deterministic chunk ID derived from file + section name, so re-ingesting
        an updated .md file overwrites the existing vector instead of duplicating it.
        """
        seed = f"{file_stem}::{section_header}".lower()
        return f"chunk_{hashlib.sha1(seed.encode()).hexdigest()[:12]}"

    def parse_markdown(self, file_path: Path) -> List[MedicalChunk]:
        """
        Parses a medical markdown file, extracts the front-matter metadata,
        peels off the H1 title, and slices the remaining content on '##' headers
        to preserve local medical context.
        """
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_metadata, clean_content = self._parse_front_matter(content)

        if "condition" not in file_metadata:
            file_metadata["condition"] = file_path.stem.replace("_", " ")
        if "source" not in file_metadata:
            file_metadata["source"] = "local_file"

        # Extract the H1 title into metadata so it doesn't become an orphan chunk
        title_match = re.match(r"^\s*#\s+(.*)", clean_content)
        if title_match:
            file_metadata["title"] = title_match.group(1).strip()
            clean_content = clean_content[title_match.end():]

        # Split remaining content on '## ' section headers
        sections = re.split(r'(?=\n## )', clean_content)
        file_chunks = []

        for section in sections:
            clean_text = section.strip()
            if not clean_text:
                continue

            header_match = re.match(r"^#+\s+(.*)", clean_text)
            section_header = header_match.group(1).strip() if header_match else "Overview"

            chunk_metadata = file_metadata.copy()
            chunk_metadata["section_header"] = section_header

            chunk_id = self._make_chunk_id(file_path.stem, section_header)

            chunk = MedicalChunk(
                chunk_id=chunk_id,
                text=clean_text,
                metadata=chunk_metadata
            )
            file_chunks.append(chunk)
            self.chunks.append(chunk)

        return file_chunks

    def get_all_formatted_chunks(self) -> List[Dict[str, Any]]:
        """
        Returns all parsed chunks structured as dictionaries, formatted
        specifically for ingestion by the MedicalVectorStore wrapper.
        """
        formatted_docs = []
        for chunk in self.chunks:
            chunk_dict = asdict(chunk)
            formatted_docs.append({
                'id': chunk_dict['chunk_id'],
                'content': chunk_dict['text'],
                'metadata': chunk_dict['metadata']
            })
        return formatted_docs

    def print_statistics(self) -> None:
        """Prints a summary of the loaded local knowledge base, including
        curation gaps (missing trust_score / last_reviewed) worth fixing."""
        if not self.chunks:
            print("⚠️ No chunks currently tracked in memory.")
            return

        print("\n" + "=" * 50)
        print("🏥 MEDIGEMMA KNOWLEDGE BASE SUMMARY")
        print("=" * 50)
        print(f"Total Chunks Generated: {len(self.chunks)}")

        conditions = set(c.metadata.get("condition", "unknown") for c in self.chunks)
        sources = set(c.metadata.get("source", "unknown") for c in self.chunks)
        categories = set(c.metadata.get("category", "unknown") for c in self.chunks)

        print(f"Unique Conditions:      {len(conditions)} ({', '.join(conditions)})")
        print(f"Document Sources:       {len(sources)} ({', '.join(sources)})")
        print(f"Categories Tracked:     {len(categories)} ({', '.join(categories)})")

        # Curation gap checks
        missing_trust = [c.metadata.get("condition", "unknown") for c in self.chunks
                          if "trust_score" not in c.metadata]
        missing_reviewed = [c.metadata.get("condition", "unknown") for c in self.chunks
                             if "last_reviewed" not in c.metadata]

        if missing_trust:
            print(f"⚠️ Missing trust_score:   {len(missing_trust)} chunks "
                  f"({', '.join(sorted(set(missing_trust)))})")
        if missing_reviewed:
            print(f"⚠️ Missing last_reviewed: {len(missing_reviewed)} chunks "
                  f"({', '.join(sorted(set(missing_reviewed)))})")
        if not missing_trust and not missing_reviewed:
            print("✅ All chunks have trust_score and last_reviewed set.")

        print("=" * 50)