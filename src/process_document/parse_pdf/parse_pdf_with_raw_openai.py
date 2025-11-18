"""
PDF Parser using raw OpenAI Vision API
Converts PDF pages to images and processes them with OpenAI's Vision API.
"""

import base64
from io import BytesIO
from pathlib import Path

from pdf2image import convert_from_path

from ...utils.api_clients import get_openai_client
from ...utils.constants import PARSE_PDF

OPENAI_CONFIG = PARSE_PDF["openai"]
OPENAI_SYSTEM_PROMPT = """Convert the following document to markdown.
Return only the markdown with no explanation text. Do not include delimiters like ```markdown or ```html.

RULES:
  - You must include all information on the page. Do not exclude headers, footers, or subtext.
  - Return tables in markdown table format (using | for columns and - for headers).
  - Charts & infographics must be interpreted to a markdown format. Prefer table format when applicable.
  - Images must be described and wrapped as: <image>description of the image</image>
  - Logos should be wrapped in brackets. Ex: <logo>Coca-Cola<logo>
  - Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY<watermark>
  - Page numbers should be wrapped in brackets. Ex: <page_number>14<page_number> or <page_number>9/22<page_number>
  - Prefer using ☐ and ☑ for check boxes."""


def _encode_image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _build_prompt(previous_markdown: str = None) -> str:
    """Build the prompt for OpenAI Vision API."""
    if previous_markdown is None:
        return OPENAI_SYSTEM_PROMPT

    format_reference = previous_markdown[: len(previous_markdown) // 2]
    return (
        f"{OPENAI_SYSTEM_PROMPT}\n\n"
        f'Markdown must maintain consistent formatting with the following page: \n\n """{format_reference}"""'
    )


def _build_request_payload(prompt: str, img_base64: str, image_detail: str) -> dict:
    """Build the request payload for OpenAI Responses API."""
    return {
        "model": OPENAI_CONFIG["model_id"],
        "reasoning": {"effort": OPENAI_CONFIG["reasoning"]},
        "input": [
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
    }


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost based on token usage."""
    return (
        input_tokens * OPENAI_CONFIG["model_input_price"]
        + output_tokens * OPENAI_CONFIG["model_output_price"]
    ) / OPENAI_CONFIG["model_pricing_unit"]


def parse_pdf_with_raw_openai(
    pdf_path: Path, image_detail: str = OPENAI_CONFIG["image_detail"]
) -> dict:
    """Parse PDF using raw OpenAI Vision API by converting to images.

    Args:
        pdf_path: Path to the PDF file
        image_detail: Detail level for images - "low" or "high"

    Returns:
        Dictionary with text, page_count, and usage information
    """
    try:
        client = get_openai_client()
        images = convert_from_path(str(pdf_path), dpi=OPENAI_CONFIG["image_dpi"])

        markdown_content = []
        total_input_tokens = 0
        total_output_tokens = 0
        previous_markdown = None

        for image in images:
            img_base64 = _encode_image_to_base64(image)
            prompt = _build_prompt(previous_markdown)
            payload = _build_request_payload(prompt, img_base64, image_detail)

            response = client.responses.create(**payload)

            page_markdown = response.output_text
            markdown_content.append(page_markdown)
            previous_markdown = page_markdown

            if hasattr(response, "usage"):
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

        full_text = "\n\n---\n\n".join(markdown_content)
        cost = _calculate_cost(total_input_tokens, total_output_tokens)

        usage = dict(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=OPENAI_CONFIG["model_id"],
            cost=cost,
        )

        return dict(text=full_text, page_count=len(images), usage=usage)

    except Exception as exc:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(exc)}") from exc
