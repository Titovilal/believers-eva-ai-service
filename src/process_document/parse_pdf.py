"""
PDF Parser Module
Functions for parsing and extracting text from PDF files using Docling.
"""

import os
import re
import base64
import logging
from io import BytesIO
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling_core.types.doc import PictureItem
from ..utils.constants import (
    PARSE_PDF_DEFAULT_IMAGE_DETAIL,
    PARSE_PDF_DOCLING_IMAGE_API_URL,
    PARSE_PDF_DOCLING_IMAGE_MODEL,
    PARSE_PDF_DOCLING_IMAGE_MAX_TOKENS,
    PARSE_PDF_DOCLING_IMAGE_PROMPT,
    PARSE_PDF_DOCLING_IMAGE_TIMEOUT,
    PARSE_PDF_DOCLING_IMAGE_TOKENS_LOW_DETAIL,
    PARSE_PDF_DOCLING_IMAGE_TOKENS_HIGH_DETAIL,
    PARSE_PDF_DOCLING_IMAGE_INPUT_PRICE,
    PARSE_PDF_DOCLING_IMAGE_OUTPUT_PRICE,
    PARSE_PDF_DOCLING_IMAGE_AVG_OUTPUT_TOKENS,
    PARSE_PDF_OPENAI_MODEL,
    PARSE_PDF_OPENAI_INPUT_PRICE,
    PARSE_PDF_OPENAI_OUTPUT_PRICE,
    PARSE_PDF_OPENAI_IMAGE_DPI,
)

logger = logging.getLogger(__name__)


# OpenAI Vision OCR prompts
SYSTEM_PROMPT_BASE = """
Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.

RULES:
  - You must include all information on the page. Do not exclude headers, footers, or subtext.
  - Return tables in markdown table format (using | for columns and - for headers).
  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
  - Images must be described and wrapped as: <image>description of the image</image>
  - Logos should be wrapped in brackets. Ex: <logo>Coca-Cola<logo>
  - Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY<watermark>
  - Page numbers should be wrapped in brackets. Ex: <page_number>14<page_number> or <page_number>9/22<page_number>
  - Prefer using ☐ and ☑ for check boxes.
"""


def _build_usage(
    input_tokens: int = 0, output_tokens: int = 0, model: str = "", cost: float = 0
) -> dict:
    """Build standardized usage dictionary."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
        "cost": cost,
    }


def _build_response(text: str, page_count: int, usage: dict) -> dict:
    """Build standardized response dictionary."""
    return {
        "text": text,
        "page_count": page_count,
        "usage": usage,
    }


def _extract_and_process_images(document, full_text: str) -> tuple[str, list[dict]]:
    """Extract and process images from document, adding annotations to text."""
    image_annotations = []

    for element, _ in document.iterate_items():
        if not isinstance(element, PictureItem):
            continue

        caption = element.caption_text(doc=document)
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
                image_annotations.append(
                    {
                        "ref": str(element.self_ref),
                        "caption": caption,
                        "annotation": ann_text,
                    }
                )
        else:
            img_parts.append("*[No description available]*")

        img_parts.append("</image>")
        full_text = full_text.replace("<!-- image -->", "\n".join(img_parts), 1)

    return full_text, image_annotations


def parse_pdf_with_docling(
    pdf_path: Path,
    enable_image_annotation: bool = False,
    image_detail: str = PARSE_PDF_DEFAULT_IMAGE_DETAIL,
) -> dict:
    """Parse PDF using Docling library with optional image annotation.

    Args:
        pdf_path: Path to the PDF file
        enable_image_annotation: Whether to enable image annotation
        image_detail: Detail level for images - "low" (70 tokens) or "high" (~630 tokens avg)
    """
    try:
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(do_ocr=False)

        if enable_image_annotation:
            pipeline_options.enable_remote_services = True
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url=PARSE_PDF_DOCLING_IMAGE_API_URL,
                params={
                    "model": PARSE_PDF_DOCLING_IMAGE_MODEL,
                    "max_completion_tokens": PARSE_PDF_DOCLING_IMAGE_MAX_TOKENS,
                },
                headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"},
                prompt=PARSE_PDF_DOCLING_IMAGE_PROMPT,
                timeout=PARSE_PDF_DOCLING_IMAGE_TIMEOUT,
            )

        # Initialize converter and process document
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))

        # Extract text and process images
        full_text = result.document.export_to_markdown()
        image_annotations = []

        if enable_image_annotation:
            full_text, image_annotations = _extract_and_process_images(
                result.document, full_text
            )

        # Extract metadata
        doc = result.document
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1

        # Calculate usage costs if image annotation is enabled
        usage = _build_usage()

        if enable_image_annotation:
            images_processed = len(image_annotations)
            # Calculate input tokens (vision tokens from images)
            tokens_per_image = (
                PARSE_PDF_DOCLING_IMAGE_TOKENS_LOW_DETAIL
                if image_detail == "low"
                else PARSE_PDF_DOCLING_IMAGE_TOKENS_HIGH_DETAIL
            )
            input_tokens = images_processed * tokens_per_image

            # Calculate output tokens (estimated average per image description)
            output_tokens = images_processed * PARSE_PDF_DOCLING_IMAGE_AVG_OUTPUT_TOKENS

            # Calculate total cost (input + output)
            cost = (
                (input_tokens * PARSE_PDF_DOCLING_IMAGE_INPUT_PRICE)
                + (output_tokens * PARSE_PDF_DOCLING_IMAGE_OUTPUT_PRICE)
            ) / 1_000_000

            usage = _build_usage(
                input_tokens, output_tokens, PARSE_PDF_DOCLING_IMAGE_MODEL, cost
            )

        return _build_response(full_text, page_count, usage)

    except Exception as e:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(e)}")


def parse_pdf_with_raw_openai(
    pdf_path: Path, image_detail: str = PARSE_PDF_DEFAULT_IMAGE_DETAIL
) -> dict:
    """Parse PDF using raw OpenAI Vision API by converting to images.

    Args:
        pdf_path: Path to the PDF file
        image_detail: Detail level for images - "low" or "high"
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=PARSE_PDF_OPENAI_IMAGE_DPI)

        markdown_content = []
        total_input_tokens = 0
        total_output_tokens = 0
        previous_markdown = None

        # Process each page/image
        for image in images:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Build the prompt based on whether we have previous context
            if previous_markdown is None:
                prompt = SYSTEM_PROMPT_BASE
            else:
                # Include consistency prompt with half of previous page
                format_reference = previous_markdown[: len(previous_markdown) // 2]
                prompt = (
                    f"{SYSTEM_PROMPT_BASE}\n\n"
                    f'Markdown must maintain consistent formatting with the following page: \n\n """{format_reference}"""'
                )

            # Call OpenAI Vision API
            response = client.responses.create(
                model=PARSE_PDF_OPENAI_MODEL,
                reasoning={"effort": "low"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{img_base64}",
                                "detail": image_detail,
                            },
                        ],
                    }
                ],
            )

            # Extract markdown content from response
            page_markdown = response.output_text
            markdown_content.append(page_markdown)

            # Save this page's markdown for next iteration
            previous_markdown = page_markdown

            # Track token usage
            if hasattr(response, "usage"):
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

        # Combine all pages
        full_text = "\n\n---\n\n".join(markdown_content)

        # Calculate costs
        cost = (
            total_input_tokens * PARSE_PDF_OPENAI_INPUT_PRICE
            + total_output_tokens * PARSE_PDF_OPENAI_OUTPUT_PRICE
        ) / 1_000_000

        usage = _build_usage(
            total_input_tokens, total_output_tokens, PARSE_PDF_OPENAI_MODEL, cost
        )

        return _build_response(full_text, len(images), usage)

    except Exception as e:
        raise Exception(f"Error parsing PDF with OpenAI {pdf_path}: {str(e)}")


def parse_pdf(
    pdf_path: str | Path,
    enable_image_annotation: bool = False,
    force_ocr: bool = False,
    image_detail: str = PARSE_PDF_DEFAULT_IMAGE_DETAIL,
) -> dict:
    """Parse a PDF file and extract all text content in a markdown format.

    Args:
        pdf_path: Path to the PDF file
        enable_image_annotation: Whether to enable image annotation (only for Docling)
        force_ocr: Whether to force OCR using OpenAI Vision API
        image_detail: Detail level for images - "low" (70 tokens) or "high" (~630 tokens)

    Returns:
        dict with keys: text, page_count, usage
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    if force_ocr:
        return parse_pdf_with_raw_openai(pdf_path, image_detail)

    return parse_pdf_with_docling(pdf_path, enable_image_annotation, image_detail)
