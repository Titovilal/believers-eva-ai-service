"""
PDF Parser Module
Functions for parsing and extracting text from PDF files using Docling.
"""
from pathlib import Path
from docling.document_converter import DocumentConverter


def parse_pdf_to_text(pdf_path: str | Path) -> dict:
    """
    Parse a PDF file and extract all text content with metadata using Docling.
    
    Args:
        pdf_path: Path to the PDF file to parse
        
    Returns:
        dict: Dictionary containing:
            - text: Extracted text from all pages
            - metadata: PDF metadata (title, author, page count, etc.)
            - page_count: Number of pages in the PDF
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a PDF
        Exception: If there's an error parsing the PDF
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    try:
        # Initialize Docling document converter
        converter = DocumentConverter()
        
        # Convert the PDF document
        result = converter.convert(str(pdf_path))
        
        # Extract text content
        full_text = result.document.export_to_markdown()
        
        # Extract metadata from the document
        doc = result.document
        pdf_metadata = {
            'title': getattr(doc, 'name', pdf_path.stem),
            'author': 'Unknown',
            'subject': 'Unknown',
            'creator': 'Unknown',
            'producer': 'Docling',
        }
        
        # Count pages (try to get from document structure)
        page_count = 0
        if hasattr(doc, 'pages') and doc.pages:
            page_count = len(doc.pages)
        else:
            # Fallback: estimate from text if pages info not available
            page_count = 1
        
        return {
            'text': full_text,
            'metadata': pdf_metadata,
            'page_count': page_count,
            'file_path': str(pdf_path),
            'file_name': pdf_path.name
        }
    
    except Exception as e:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(e)}")