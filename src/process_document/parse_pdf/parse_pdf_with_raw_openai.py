"""
PDF Parser using raw OpenAI Vision API
Converts PDF pages to images and processes them with OpenAI's Vision API.
"""

import base64
from io import BytesIO
from pathlib import Path
from pdf2image import convert_from_path
from ...utils.constants import (
    PARSE_PDF_DEFAULT_IMAGE_DETAIL,
    PARSE_PDF_OPENAI_MODEL,
    PARSE_PDF_OPENAI_INPUT_PRICE,
    PARSE_PDF_OPENAI_OUTPUT_PRICE,
    PARSE_PDF_OPENAI_IMAGE_DPI,
)
from ...utils.api_clients import get_openai_client


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


def parse_pdf_with_raw_openai(
    pdf_path: Path, image_detail: str = PARSE_PDF_DEFAULT_IMAGE_DETAIL
) -> dict:
    """Parse PDF using raw OpenAI Vision API by converting to images.

    Args:
        pdf_path: Path to the PDF file
        image_detail: Detail level for images - "low" or "high"
    """
    try:
        # Get OpenAI client from singleton
        client = get_openai_client()

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

        usage = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "model": PARSE_PDF_OPENAI_MODEL,
            "cost": cost,
        }
        return dict(text=full_text, page_count=len(images), usage=usage)

    except Exception as e:
        raise Exception(f"Error parsing PDF with OpenAI {pdf_path}: {str(e)}")
