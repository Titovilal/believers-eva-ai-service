"""
PDF Parser Module
Functions for parsing and extracting text from PDF files using Docling.
"""

import os
import re
import base64
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
    IMAGE_ANNOTATION_API_URL,
    IMAGE_ANNOTATION_MODEL,
    IMAGE_ANNOTATION_MAX_TOKENS,
    IMAGE_ANNOTATION_PROMPT,
    IMAGE_ANNOTATION_TIMEOUT,
)


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


def _consistency_prompt(prior_page: str) -> str:
    """Generate consistency prompt with prior page reference."""
    return f'Markdown must maintain consistent formatting with the following page: \n\n """{prior_page}"""'


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
    pdf_path: Path, enable_image_annotation: bool = False
) -> dict:
    """Parse PDF using Docling library with optional image annotation."""
    try:
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(do_ocr=False)

        if enable_image_annotation:
            pipeline_options.enable_remote_services = True
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url=IMAGE_ANNOTATION_API_URL,
                params={
                    "model": IMAGE_ANNOTATION_MODEL,
                    "max_completion_tokens": IMAGE_ANNOTATION_MAX_TOKENS,
                },
                headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"},
                prompt=IMAGE_ANNOTATION_PROMPT,
                timeout=IMAGE_ANNOTATION_TIMEOUT,
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

        # Extract metadata and build response
        doc = result.document
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1

        # Calculate usage costs (Vision API: ~$0.15/1M tokens, ~500 tokens/image)
        images_processed = len(image_annotations)
        vision_tokens = images_processed * 500
        vision_cost = vision_tokens * 0.15 / 1_000_000

        response = {
            "text": full_text,
            "page_count": page_count,
            "file_name": pdf_path.name,
            "usage": {
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


def parse_pdf_with_raw_openai(pdf_path: Path) -> dict:
    """Parse PDF using raw OpenAI Vision API by converting to images."""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=200)

        markdown_content = []
        total_tokens = 0
        previous_markdown = None

        # Process each page/image
        for image in images:
            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Build the prompt based on whether we have previous context
            if previous_markdown is None:
                # First page: base prompt only
                prompt = SYSTEM_PROMPT_BASE
            else:
                # Subsequent pages: include consistency prompt with half of previous page
                half_length = len(previous_markdown) // 2
                format_reference = previous_markdown[:half_length]
                prompt = (
                    SYSTEM_PROMPT_BASE + "\n\n" + _consistency_prompt(format_reference)
                )

            # Call OpenAI Vision API
            response = client.responses.create(
                model="gpt-5-nano",
                reasoning={"effort": "low"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{img_base64}"
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
                total_tokens += response.usage.total_tokens

        # Combine all pages
        full_text = "\n\n---\n\n".join(markdown_content)

        # Calculate costs (gpt-4o: ~$2.50/1M input tokens, ~$10/1M output tokens)
        # Vision tokens are typically higher, estimate ~1000 tokens per image input
        estimated_cost = total_tokens * 5 / 1_000_000  # Average rate

        return {
            "text": full_text,
            "page_count": len(images),
            "file_name": pdf_path.name,
            "usage": {
                "total_tokens": total_tokens,
                "pages_processed": len(images),
                "estimated_cost": estimated_cost,
            },
        }

    except Exception as e:
        raise Exception(f"Error parsing PDF with OpenAI {pdf_path}: {str(e)}")


def parse_pdf(
    pdf_path: str | Path, enable_image_annotation: bool = False, force_ocr: bool = False
) -> dict:
    """Parse a PDF file and extract all text content in a markdown format"""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    if force_ocr:
        return parse_pdf_with_raw_openai(pdf_path)

    return parse_pdf_with_docling(pdf_path, enable_image_annotation)
