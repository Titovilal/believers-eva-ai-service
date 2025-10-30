"""
PDF Parser Module
Functions for parsing and extracting text from PDF files using Docling.
"""
import os
import re
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling_core.types.doc import PictureItem


def _get_openai_vlm_options():
    """
    Configure OpenAI Vision Language Model options for image annotation.

    Requires environment variable:
        - OPENAI_API_KEY: OpenAI API key
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for image annotation")

    options = PictureDescriptionApiOptions(
        url="https://api.openai.com/v1/chat/completions",
        params=dict(
            model="gpt-4o",
            max_tokens=300,
        ),
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        prompt="Describe the image in three sentences. Be concise and accurate.",
        timeout=60,
    )
    return options


def parse_pdf_to_text(pdf_path: str | Path, enable_image_annotation: bool = False) -> dict:
    """
    Parse a PDF file and extract all text content with metadata using Docling.

    Args:
        pdf_path: Path to the PDF file to parse
        enable_image_annotation: If True, annotate images with AI-generated descriptions.
                                Requires OPENAI_API_KEY environment variable

    Returns:
        dict: Dictionary containing:
            - text: Extracted text from all pages (with image annotations if enabled)
            - metadata: PDF metadata (title, author, page count, etc.)
            - page_count: Number of pages in the PDF
            - images: List of image annotations (if enabled)

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a PDF or OPENAI_API_KEY is missing
        Exception: If there's an error parsing the PDF
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    try:
        # Configure pipeline options for image annotation if enabled
        if enable_image_annotation:
            pipeline_options = PdfPipelineOptions(
                enable_remote_services=True  # Required for picture descriptions
            )
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = _get_openai_vlm_options()

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        else:
            # Initialize standard Docling document converter
            converter = DocumentConverter()

        # Convert the PDF document
        result = converter.convert(str(pdf_path))

        # Extract text content and process images
        image_annotations = []
        full_text = result.document.export_to_markdown()

        if enable_image_annotation:
            # Replace image placeholders with annotated descriptions
            for element, _ in result.document.iterate_items():
                if isinstance(element, PictureItem):
                    caption = element.caption_text(doc=result.document)
                    annotations = element.annotations

                    # Create annotated image section with <image> tags
                    img_text = "<image>\n"
                    if caption:
                        img_text += f"**Caption:** {caption}\n\n"
                    if annotations:
                        img_text += "**Description:**\n"
                        for annotation in annotations:
                            # Extract just the text content from the annotation
                            annotation_text = str(annotation)
                            if 'text=' in annotation_text:
                                # Parse the annotation to get just the description text
                                match = re.search(r'text=["\'](.+?)["\']', annotation_text)
                                if match:
                                    annotation_text = match.group(1)
                            img_text += f"{annotation_text}\n"
                            image_annotations.append({
                                'ref': str(element.self_ref),
                                'caption': caption,
                                'annotation': annotation_text
                            })
                    else:
                        img_text += "*[No description available]*\n"
                    img_text += "</image>"

                    # Replace the placeholder in the markdown
                    full_text = full_text.replace("<!-- image -->", img_text, 1)
        
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
        
        result_dict = {
            'text': full_text,
            'metadata': pdf_metadata,
            'page_count': page_count,
            'file_path': str(pdf_path),
            'file_name': pdf_path.name
        }

        # Add image annotations if enabled
        if enable_image_annotation:
            result_dict['images'] = image_annotations

        return result_dict
    
    except Exception as e:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(e)}")