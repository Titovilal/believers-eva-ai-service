"""
Execute PDF Parser
Script to demonstrate PDF parsing functionality.
"""

from base_cookbook import BaseCookbook
from src.rag.parse_pdf import parse_pdf


class PDFParserCookbook(BaseCookbook):
    """Cookbook for PDF parsing functionality."""

    def __init__(self):
        super().__init__("PDF PARSER EXECUTION")

    def run(self):
        # Path to the sample PDF
        pdf_path = self.data_dir / "sample.pdf"

        self.print_header(f"Parsing PDF: {pdf_path.name}")

        # Parse PDF to text
        result = parse_pdf(pdf_path)

        # Display metadata
        print("\n📄 PDF METADATA:")
        print(f"  File Name: {result['file_name']}")
        print(f"  Page Count: {result['page_count']}")
        for key, value in result["metadata"].items():
            print(f"  {key.title()}: {value}")

        # Display text content
        print("\n📝 EXTRACTING TEXT CONTENT:")
        print("-" * 80)
        text_content = result["text"]

        # Display first 2000 characters as preview
        if len(text_content) > 2000:
            print(text_content[:2000])
            print(f"\n... (truncated - total length: {len(text_content)} characters)")
        else:
            print(text_content)

        # Save to file
        output_path = self.save_text_file(text_content, f"{pdf_path.stem}.md")

        self.print_success("Successfully parsed PDF!", output_path)


if __name__ == "__main__":
    cookbook = PDFParserCookbook()
    cookbook.execute()
