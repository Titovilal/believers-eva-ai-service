"""
Execute Chunking
Script to demonstrate text chunking functionality.
"""
from base_cookbook import BaseCookbook
from src.rag.generate_chunks import generate_chunks


class ChunkingCookbook(BaseCookbook):
    """Cookbook for text chunking."""

    def __init__(self):
        super().__init__("CHUNKING GENERATION")

    def run(self):
        # Path to the sample markdown file
        md_path = self.data_dir / "sample.md"

        self.print_header(f"Processing file: {md_path.name}")

        # Step 1: Read markdown file
        print("\n📄 Step 1: Reading markdown file...")
        text_content = self.read_file(md_path)
        print(f"  ✓ Read {len(text_content)} characters")

        # Step 2: Generate chunks
        print("\n🔪 Step 2: Generating chunks...")
        chunks = generate_chunks(
            text=text_content,
        )
        print(f"  ✓ Generated {len(chunks)} chunks")

        # Display sample chunks
        print("\n📦 Sample chunks (first 3):")
        print("-" * 80)
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(f"\nChunk {i} ({len(chunk)} chars):")
            print(preview)

        if len(chunks) > 3:
            print(f"\n... (showing 3 of {len(chunks)} chunks)")

        # Save chunks as text file
        chunks_content = ""
        for i, chunk in enumerate(chunks, 1):
            chunks_content += f"{'=' * 80}\n"
            chunks_content += f"CHUNK {i}\n"
            chunks_content += f"{'=' * 80}\n"
            chunks_content += chunk
            chunks_content += "\n\n"

        chunks_path = self.save_text_file(chunks_content, f"{md_path.stem}_chunks.txt")

        print("\n" + "=" * 80)
        print("✅ Successfully generated chunks!")
        print(f"📁 Chunks saved to: {chunks_path}")
        print("=" * 80)


if __name__ == "__main__":
    cookbook = ChunkingCookbook()
    cookbook.execute()
