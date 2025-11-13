"""
Execute PDF Parser
Script to demonstrate PDF parsing functionality.
"""

from base_cookbook import BaseCookbook
from src.process_document.parse_pdf import parse_pdf


class PDFParserCookbook(BaseCookbook):
    """Cookbook for PDF parsing functionality."""

    def __init__(self):
        super().__init__("PDF PARSER EXECUTION")

    def _display_result(self, result: dict, method_name: str):
        """Display parsing result with usage information."""
        text_content = result["text"]

        print(f"\n📝 EXTRACTED TEXT ({method_name}):")
        print("-" * 80)

        # Display first 100 characters as preview
        if len(text_content) > 100:
            print(text_content[:100])
            print(f"\n... (truncated - total length: {len(text_content)} characters)")
        else:
            print(text_content)

        # Display usage information
        print(f"\n📊 USAGE INFORMATION ({method_name}):")
        print("-" * 80)
        usage = result.get("usage", {})
        for key, value in usage.items():
            # Format the key to be more readable
            readable_key = key.replace("_", " ").title()
            if value is None:
                print(f"{readable_key}: N/A")
            elif "cost" in key.lower():
                print(f"{readable_key}: ${value:.6f}")
            else:
                print(f"{readable_key}: {value}")
        print()

    def run(self):
        # Path to the sample PDF
        pdf_path = self.data_dir / "sample.pdf"

        self.print_header(f"Parsing PDF: {pdf_path.name}")
        print("\nTesting 3 different parsing methods:\n")

        # Method 1: Docling without image annotation
        print("\n" + "=" * 80)
        print("METHOD 1: DOCLING (Standard)")
        print("=" * 80)
        result1 = parse_pdf(pdf_path, force_ocr=False, enable_image_annotation=False)
        self._display_result(result1, "Docling Standard")
        output_path1 = self.save_text_file(
            result1["text"], f"{pdf_path.stem}_docling.md"
        )
        print(f"✓ Saved to: {output_path1}")

        # Method 2: Docling with image annotation
        print("\n" + "=" * 80)
        print("METHOD 2: DOCLING (With Image Annotation)")
        print("=" * 80)
        result2 = parse_pdf(pdf_path, force_ocr=False, enable_image_annotation=True)
        self._display_result(result2, "Docling + Image Annotation")
        output_path2 = self.save_text_file(
            result2["text"], f"{pdf_path.stem}_docling_annotated.md"
        )
        print(f"✓ Saved to: {output_path2}")

        # Method 3: Force OCR (GPT-5)
        print("\n" + "=" * 80)
        print("METHOD 3: GPT-5 OCR")
        print("=" * 80)
        result3 = parse_pdf(pdf_path, force_ocr=True)
        self._display_result(result3, "GPT-5 OCR")
        output_path3 = self.save_text_file(result3["text"], f"{pdf_path.stem}_gpt5.md")
        print(f"✓ Saved to: {output_path3}")

        self.print_success("Successfully tested all 3 parsing methods!")


if __name__ == "__main__":
    cookbook = PDFParserCookbook()
    cookbook.execute()
