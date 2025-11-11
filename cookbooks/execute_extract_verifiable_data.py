"""
Execute Verifiable Data Extraction
Script to demonstrate verifiable data extraction functionality.
"""

import asyncio
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

    async def run_async(self):
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

        print(f"\n📊 TOTAL CHUNKS: {len(sample_chunks)}")
        print(f"  Chunks with numbers: {sum(chunks_with_numbers)}")

        # Filter to only process chunks that contain numbers and keep track of original indices
        filtered_chunks = []
        original_indices = []
        for i, chunk in enumerate(sample_chunks):
            if chunks_with_numbers[i]:
                filtered_chunks.append(chunk)
                original_indices.append(i)

        filtered_chunks_with_numbers = [True] * len(filtered_chunks)

        print(f"  Chunks to process (only with numbers): {len(filtered_chunks)}")
        print("-" * 80)

        # Extract verifiable data only from chunks with numbers
        extraction_result = await extract_verifiable_data(
            chunks=filtered_chunks,
            chunks_with_numbers=filtered_chunks_with_numbers,
        )

        # Restore original chunk indices
        verifiable_facts_unfiltered = []
        for fact_group in extraction_result["verifiable_facts"]:
            restored_fact_group = fact_group.copy()
            restored_fact_group["chunk_index"] = original_indices[
                fact_group["chunk_index"]
            ]
            verifiable_facts_unfiltered.append(restored_fact_group)

        # Filter statements to only include those with numbers
        verifiable_facts_filtered = []
        for fact_group in verifiable_facts_unfiltered:
            if "statements" in fact_group and fact_group["statements"]:
                # Filter statements that contain numbers
                statements_with_numbers = [
                    stmt
                    for stmt in fact_group["statements"]
                    if detect_number_in_text(stmt, lang="en")
                ]

                if statements_with_numbers:
                    filtered_fact_group = fact_group.copy()
                    filtered_fact_group["statements"] = statements_with_numbers
                    verifiable_facts_filtered.append(filtered_fact_group)

        # Build result with both versions
        result = {
            "summary": {
                "total_chunks_analyzed": len(filtered_chunks),
                "total_statements_extracted_unfiltered": sum(
                    len(fg["statements"]) for fg in verifiable_facts_unfiltered
                ),
                "total_statements_extracted_filtered": sum(
                    len(fg["statements"]) for fg in verifiable_facts_filtered
                ),
            },
            "verifiable_facts_unfiltered": verifiable_facts_unfiltered,
            "verifiable_facts_filtered": verifiable_facts_filtered,
        }

        # Display results
        self.print_section("📋 EXTRACTION RESULTS")

        print("\nSummary:")
        print(f"  Total chunks analyzed: {result['summary']['total_chunks_analyzed']}")
        print(
            f"  Total statements extracted (unfiltered): {result['summary']['total_statements_extracted_unfiltered']}"
        )
        print(
            f"  Total statements extracted (filtered): {result['summary']['total_statements_extracted_filtered']}"
        )

        print(
            "\n🔍 Verifiable facts by chunk (FILTERED - only statements with numbers):"
        )
        print("-" * 80)

        for fact_group in result["verifiable_facts_filtered"]:
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

    def run(self):
        """Synchronous wrapper for async run method."""
        asyncio.run(self.run_async())


if __name__ == "__main__":
    cookbook = VerifiableDataCookbook()
    cookbook.execute()
