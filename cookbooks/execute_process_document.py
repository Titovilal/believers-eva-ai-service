"""
Execute Process Document
Script to demonstrate end-to-end document processing with RAG pipeline.
"""

import base64
from base_cookbook import BaseCookbook
from src.process_document.process_document import process_document


class ProcessDocumentCookbook(BaseCookbook):
    """Cookbook for full document processing pipeline."""

    def __init__(self):
        super().__init__("PROCESS DOCUMENT EXECUTION")

    def run(self):
        # Path to the sample PDF
        pdf_path = self.data_dir / "sample.pdf"

        self.print_header(f"Processing Document: {pdf_path.name}")

        # Read and encode the PDF to base64
        print("\n📖 Reading and encoding document...")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            base64_data = base64.b64encode(pdf_bytes).decode("utf-8")

        # Process the document
        print("🔄 Processing document (parsing, chunking, embeddings, verifiable data)...")
        result = process_document(base64_data)

        # Display basic info
        print("\n📄 DOCUMENT INFO:")
        print(f"  File Type: {result['file_type']}")
        if result["file_type"] == "pdf":
            print(f"  File Name: {result['file_name']}")
            print(f"  Page Count: {result['page_count']}")

        # Display text preview
        self.print_section("📝 TEXT CONTENT PREVIEW")
        text_content = result["text"]
        if len(text_content) > 500:
            print(text_content[:500])
            print(f"\n... (truncated - total length: {len(text_content)} characters)")
        else:
            print(text_content)

        # Display chunks info
        self.print_section("🧩 CHUNKS")
        chunks = result["chunks"]
        print(f"  Total Chunks: {len(chunks)}")
        print(f"  Chunks with Numbers: {sum(result['chunks_with_numbers'])}")
        print("\n  First 3 chunks preview:")
        for i, chunk in enumerate(chunks[:3], 1):
            has_number = "✓" if result["chunks_with_numbers"][i - 1] else "✗"
            preview = chunk[:100].replace("\n", " ")
            if len(chunk) > 100:
                preview += "..."
            print(f"    {i}. [{has_number}] {preview}")

        # Display embeddings info
        self.print_section("🔢 EMBEDDINGS")
        embeddings = result["embeddings"]
        print(f"  Total Embeddings: {len(embeddings)}")
        print(f"  Embedding Dimension: {len(embeddings[0]) if embeddings else 0}")
        if embeddings:
            print(f"  First embedding preview: [{embeddings[0][0]:.6f}, {embeddings[0][1]:.6f}, ...]")

        # Display verifiable facts
        if "verifiable_facts" in result:
            self.print_section("✅ VERIFIABLE FACTS")
            verifiable = result["verifiable_facts"]
            print(
                f"  Total Chunks Analyzed: {verifiable['summary']['total_chunks_analyzed']}"
            )
            print(
                f"  Total Statements Extracted: {verifiable['summary']['total_statements_extracted']}"
            )

            print("\n  Extracted Statements:")
            for item in verifiable["verifiable_facts"]:
                chunk_idx = item["chunk_index"]
                statements = item.get("statements", [])
                if statements:
                    print(f"\n  Chunk {chunk_idx}:")
                    for stmt in statements:
                        print(f"    • {stmt}")

        # Save full result to JSON
        output_json = self.save_json_file(result, f"{pdf_path.stem}_processed.json")

        # Save just verifiable facts to separate file
        if "verifiable_facts" in result:
            verifiable_json = self.save_json_file(
                result["verifiable_facts"], f"{pdf_path.stem}_verifiable_facts.json"
            )
            print(f"\n📁 Verifiable facts saved to: {verifiable_json}")

        self.print_success("Successfully processed document!", output_json)


if __name__ == "__main__":
    cookbook = ProcessDocumentCookbook()
    cookbook.execute()
