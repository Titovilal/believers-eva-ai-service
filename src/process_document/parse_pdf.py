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
from ..utils.constants import (
    IMAGE_ANNOTATION_API_URL,
    IMAGE_ANNOTATION_MODEL,
    IMAGE_ANNOTATION_MAX_TOKENS,
    IMAGE_ANNOTATION_PROMPT,
    IMAGE_ANNOTATION_TIMEOUT,
)


def _get_openai_vlm_options():
    """
    Configure OpenAI Vision Language Model options for image annotation.

    Requires environment variable:
        - OPENAI_API_KEY: OpenAI API key

    Raises:
        ValueError: If OPENAI_API_KEY is missing or model is not gpt-5-nano
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for image annotation"
        )

    if IMAGE_ANNOTATION_MODEL != "gpt-5-nano":
        raise ValueError(
            f"Only 'gpt-5-nano' model is supported for image annotation, got '{IMAGE_ANNOTATION_MODEL}'"
        )

    return PictureDescriptionApiOptions(
        url=IMAGE_ANNOTATION_API_URL,
        params={
            "model": IMAGE_ANNOTATION_MODEL,
            "max_completion_tokens": IMAGE_ANNOTATION_MAX_TOKENS,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        prompt=IMAGE_ANNOTATION_PROMPT,
        timeout=IMAGE_ANNOTATION_TIMEOUT,
    )


def parse_pdf(
    pdf_path: str | Path, enable_image_annotation: bool = False, force_ocr: bool = False
) -> dict:
    """
    Parse a PDF file and extract all text content with metadata using Docling.

    Args:
        pdf_path: Path to the PDF file to parse
        enable_image_annotation: If True, annotate images with AI-generated descriptions.
                                Requires OPENAI_API_KEY environment variable
        force_ocr: If True, force OCR even for PDFs with native text.
                  If False (default), prefer native text extraction (faster)

    Returns:
        dict: Dictionary containing:
            - text: Extracted text from all pages (with image annotations if enabled)
            - metadata: PDF metadata (title, author, page count, etc.)
            - page_count: Number of pages in the PDF
            - images: List of image annotations (if enabled)
            - usage: Processing costs and metrics

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a PDF or OPENAI_API_KEY is missing
        Exception: If there's an error parsing the PDF
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    try:
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(do_ocr=force_ocr)

        if enable_image_annotation:
            pipeline_options.enable_remote_services = True
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = _get_openai_vlm_options()

        # Initialize converter and process document
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))

        # Extract text and process images
        image_annotations = []
        full_text = result.document.export_to_markdown()

        if enable_image_annotation:
            for element, _ in result.document.iterate_items():
                if not isinstance(element, PictureItem):
                    continue

                caption = element.caption_text(doc=result.document)
                annotations = element.annotations

                # Build image section
                img_parts = ["<image>"]
                if caption:
                    img_parts.append(f"**Caption:** {caption}\n")

                if annotations:
                    img_parts.append("**Description:**")
                    for annotation in annotations:
                        # Extract text from annotation
                        ann_text = str(annotation)
                        if match := re.search(r'text=["\'](.+?)["\']', ann_text):
                            ann_text = match.group(1)

                        img_parts.append(ann_text)
                        image_annotations.append({
                            "ref": str(element.self_ref),
                            "caption": caption,
                            "annotation": ann_text,
                        })
                else:
                    img_parts.append("*[No description available]*")

                img_parts.append("</image>")
                full_text = full_text.replace("<!-- image -->", "\n".join(img_parts), 1)

        # Extract metadata and build response
        doc = result.document
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1

        # Calculate usage costs (Vision API: ~$0.15/1M tokens, ~500 tokens/image)
        images_processed = len(image_annotations)
        vision_tokens = images_processed * 500
        vision_cost = vision_tokens * 0.15 / 1_000_000

        response = {
            "text": full_text,
            "metadata": {
                "title": getattr(doc, "name", pdf_path.stem),
                "author": "Unknown",
                "subject": "Unknown",
                "creator": "Unknown",
                "producer": "Docling",
            },
            "page_count": page_count,
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "usage": {
                "ocr_used": force_ocr,
                "images_annotated": images_processed,
                "vision_tokens_estimated": vision_tokens,
                "vision_cost": vision_cost,
            },
        }

        if enable_image_annotation:
            response["images"] = image_annotations

        return response

    except Exception as e:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(e)}")
