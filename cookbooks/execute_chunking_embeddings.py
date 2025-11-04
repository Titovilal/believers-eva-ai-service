"""
Execute Chunking and Embeddings
Script to demonstrate text chunking and embeddings generation functionality.
"""
from base_cookbook import BaseCookbook
from src.rag.db import generate_chunks_from_text, generate_embeddings


class ChunkingEmbeddingsCookbook(BaseCookbook):
    """Cookbook for text chunking and embeddings generation."""

    def __init__(self):
        super().__init__("CHUNKING AND EMBEDDINGS GENERATION")

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
        chunks = generate_chunks_from_text(
            text=text_content,
            chunk_size=1024,
            chunk_overlap=0,
            model="text-embedding-3-small"
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

        # Step 3: Generate embeddings
        print("\n🧮 Step 3: Generating embeddings...")
        print(f"  Processing {len(chunks)} chunks...")
        embeddings = generate_embeddings(
            chunks=chunks,
            model="text-embedding-3-small"
        )
        print(f"  ✓ Generated {len(embeddings)} embeddings")
        print(f"  ✓ Embedding dimension: {len(embeddings[0])}")

        # Display sample embedding
        print("\n🔢 Sample embedding (first 10 dimensions of first chunk):")
        print(f"  {embeddings[0][:10]}...")

        # Save chunks as text file
        chunks_content = ""
        for i, chunk in enumerate(chunks, 1):
            chunks_content += f"{'=' * 80}\n"
            chunks_content += f"CHUNK {i}\n"
            chunks_content += f"{'=' * 80}\n"
            chunks_content += chunk
            chunks_content += "\n\n"

        chunks_path = self.save_text_file(chunks_content, f"{md_path.stem}_chunks.txt")

        # Save embeddings as JSON
        embeddings_data = {
            'chunk_count': len(chunks),
            'embedding_dimension': len(embeddings[0]),
            'embeddings': embeddings
        }
        embeddings_path = self.save_json_file(embeddings_data, f"{md_path.stem}_embeddings.json")

        print("\n" + "=" * 80)
        print("✅ Successfully generated chunks and embeddings!")
        print(f"📁 Chunks saved to: {chunks_path}")
        print(f"📁 Embeddings saved to: {embeddings_path}")
        print("=" * 80)


if __name__ == "__main__":
    cookbook = ChunkingEmbeddingsCookbook()
    cookbook.execute()
