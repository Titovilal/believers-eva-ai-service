"""
Execute Verifiable Data Extraction
Script to demonstrate verifiable data extraction functionality.
"""

import re
from base_cookbook import BaseCookbook
from src.process_document.extract_verifiable_data import extract_verifiable_data
from src.process_document.detect_number_in_text import detect_number_in_text


class VerifiableDataCookbook(BaseCookbook):
    """Cookbook for verifiable data extraction functionality."""

    def __init__(self):
        super().__init__("VERIFIABLE DATA EXTRACTION")

    def parse_chunks_from_file(self, file_path):
        """
        Parse chunks from a text file where chunks are separated by separator lines.

        Args:
            file_path: Path to the file containing chunks

        Returns:
            list: List of chunk texts
        """
        content = self.read_file(file_path)

        # Split by chunk separator pattern
        # Pattern: ====... followed by CHUNK N followed by ====...
        chunk_pattern = r"={80,}\s*CHUNK \d+\s*={80,}\s*"
        chunks = re.split(chunk_pattern, content)

        # Remove empty strings and strip whitespace
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        return chunks

    def run(self):
        # Path to the sample chunks file
        chunks_file = self.data_dir / "sample_chunks.txt"

        self.print_header(f"Reading chunks from: {chunks_file.name}")

        # Parse chunks from file
        sample_chunks = self.parse_chunks_from_file(chunks_file)

        print(f"\n📄 Loaded {len(sample_chunks)} chunks from file")

        # Identify chunks with numbers using detect_number_in_text
        chunks_with_numbers = [
            detect_number_in_text(chunk, lang="en") for chunk in sample_chunks
        ]

        self.print_header("Analyzing text chunks for verifiable data...")

        print(f"\n📊 PROCESSING {len(sample_chunks)} TEXT CHUNKS:")
        print(f"  Chunks with numbers: {sum(chunks_with_numbers)}")
        print("-" * 80)

        # Extract verifiable data
        result = extract_verifiable_data(
            chunks=sample_chunks,
            chunks_with_numbers=chunks_with_numbers,
        )

        # Reorder result to put summary first in JSON
        result = {
            "summary": result["summary"],
            "verifiable_facts": result["verifiable_facts"]
        }

        # Display results
        self.print_section("📋 EXTRACTION RESULTS")

        print("\nSummary:")
        print(f"  Total chunks analyzed: {result['summary']['total_chunks_analyzed']}")
        print(f"  Total statements extracted: {result['summary']['total_statements_extracted']}")

        print("\n🔍 Verifiable facts by chunk:")
        print("-" * 80)

        for fact_group in result["verifiable_facts"]:
            chunk_idx = fact_group["chunk_index"]
            chunk_text = sample_chunks[chunk_idx]
            print(f"\n📄 Chunk #{chunk_idx}:")

            # Show preview of chunk (first 150 chars)
            preview = chunk_text[:150].replace("\n", " ")
            if len(chunk_text) > 150:
                preview += "..."
            print(f"   Original: {preview}")

            if "error" in fact_group:
                print(f"   ❌ Error: {fact_group['error']}")
            else:
                statements = fact_group.get("statements", [])
                if statements:
                    print(f"   ✅ Extracted {len(statements)} statement(s):")
                    for i, statement in enumerate(statements, 1):
                        print(f"      {i}. {statement}")
                else:
                    print("   ℹ️  No verifiable data found")

        # Save results to JSON with input filename
        output_filename = f"{chunks_file.stem}_verifiable_data.json"
        output_path = self.save_json_file(result, output_filename)

        self.print_success("Successfully extracted verifiable data!", output_path)


if __name__ == "__main__":
    cookbook = VerifiableDataCookbook()
    cookbook.execute()
