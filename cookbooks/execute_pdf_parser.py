"""
Execute PDF Parser
Script to demonstrate PDF parsing functionality.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.pdf_parser import parse_pdf_to_text  # noqa: E402


def main():
    # Path to the sample PDF
    pdf_path = Path(__file__).parent.parent / "data" / "sample_pdfs" / "somatosensory.pdf"
    
    print("=" * 80)
    print("PDF PARSER EXECUTION")
    print("=" * 80)
    print(f"\nParsing PDF: {pdf_path.name}")
    print("-" * 80)
    
    try:
        # Parse PDF to text
        result = parse_pdf_to_text(pdf_path)
        
        # Display metadata
        print("\n📄 PDF METADATA:")
        print(f"  File Name: {result['file_name']}")
        print(f"  Page Count: {result['page_count']}")
        for key, value in result['metadata'].items():
            print(f"  {key.title()}: {value}")
        
        # Display text content
        print("\n📝 EXTRACTING TEXT CONTENT:")
        print("-" * 80)
        text_content = result['text']
        
        # Display first 2000 characters as preview
        if len(text_content) > 2000:
            print(text_content[:2000])
            print(f"\n... (truncated - total length: {len(text_content)} characters)")
        else:
            print(text_content)
        
        # Save to file in cookbooks/outputs directory
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print("\n" + "=" * 80)
        print("✅ Successfully parsed PDF!")
        print(f"📁 Full text saved to: {output_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
