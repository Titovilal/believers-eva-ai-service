"""
PDF Parser using Docling library
Uses Docling for document conversion with optional image annotation.
"""

import re
import base64
import threading
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image

from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, VlmStopReason
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
import docling.models.picture_description_api_model as model_module
import docling.utils.api_image_request as api_module
from docling_core.types.doc import PictureItem

from ...utils.logs import log_exception

from ...utils.api_clients import get_openai_client
from ...utils.constants import PARSE_PDF

DOCLING_CONFIG = PARSE_PDF["docling"]
DOCLING_SYSTEM_PROMPT = "Describe the picture."


class UsageTracker:
    """Context-bound tracker to isolate usage accounting per request."""

    _context: ContextVar[dict] = ContextVar(
        "_usage_context", default={"input_tokens": 0, "output_tokens": 0}
    )
    _lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        cls._context.set({"input_tokens": 0, "output_tokens": 0})

    @classmethod
    def record(cls, input_tokens: int, output_tokens: int) -> None:
        with cls._lock:
            current = cls._context.get()
            cls._context.set(
                {
                    "input_tokens": current["input_tokens"] + input_tokens,
                    "output_tokens": current["output_tokens"] + output_tokens,
                }
            )

    @classmethod
    def snapshot(cls) -> dict:
        return cls._context.get().copy()


def _encode_image_to_base64(image: Image.Image) -> str:
    img_io = BytesIO()
    image.save(img_io, "PNG")
    return base64.b64encode(img_io.getvalue()).decode("utf-8")


def _build_responses_request_payload(
    prompt: str, image_base64: str, params: dict
) -> dict:
    return {
        "model": params.get("model"),
        "reasoning": params.get("reasoning"),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                        "detail": params.get("detail"),
                    },
                ],
            }
        ],
    }


def responses_api_image_request(
    image: Image.Image,
    prompt: str,
    **params,
) -> Tuple[str, Optional[int], VlmStopReason]:
    """Custom API request function for OpenAI Responses API using SDK."""
    client = get_openai_client()
    image_base64 = _encode_image_to_base64(image)
    payload = _build_responses_request_payload(prompt, image_base64, params)

    try:
        response = client.responses.create(**payload)
        generated_text = response.output_text.strip()
        num_tokens = response.usage.input_tokens + response.usage.output_tokens
        stop_reason = VlmStopReason.END_OF_SEQUENCE

        UsageTracker.record(response.usage.input_tokens, response.usage.output_tokens)

        return generated_text, num_tokens, stop_reason

    except Exception as exc:
        log_exception("Error annotating image description", exc)
        raise


def _build_usage(
    input_tokens: int = 0, output_tokens: int = 0, model: str = "", cost: float = 0
) -> dict:
    return dict(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cost=cost,
    )


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


def _build_pipeline_options(
    enable_image_annotation: bool, image_detail: str
) -> PdfPipelineOptions:
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_V2),
    )

    if not enable_image_annotation:
        return pipeline_options

    pipeline_options.enable_remote_services = True
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        params=dict(
            model=DOCLING_CONFIG["model_id"],
            reasoning={"effort": DOCLING_CONFIG["reasoning"]},
            detail=image_detail,
        ),
        prompt=DOCLING_SYSTEM_PROMPT,
        concurrency=DOCLING_CONFIG["concurrency"],
    )

    return pipeline_options


def _calculate_usage(enable_image_annotation: bool) -> dict:
    if not enable_image_annotation:
        return _build_usage()

    usage_data = UsageTracker.snapshot()
    input_tokens = usage_data["input_tokens"]
    output_tokens = usage_data["output_tokens"]
    print("-" * 50)
    print("Docling Usage:")
    print(f"  Input Tokens: {input_tokens}")
    print(f"  Output Tokens: {output_tokens}")
    print("-" * 50)


    cost = (
        input_tokens * DOCLING_CONFIG["model_input_price"]
        + output_tokens * DOCLING_CONFIG["model_output_price"]
    ) / DOCLING_CONFIG["model_pricing_unit"]

    return _build_usage(
        input_tokens,
        output_tokens,
        DOCLING_CONFIG["model_id"],
        cost,
    )


def parse_pdf_with_docling(
    pdf_path: Path,
    enable_image_annotation: bool = False,
    image_detail: str = DOCLING_CONFIG["image_detail"],
) -> dict:
    """Parse PDF using Docling library with optional image annotation.

    Args:
        pdf_path: Path to the PDF file
        enable_image_annotation: Whether to enable image annotation
        image_detail: Detail level for images - "low" (70 tokens) or "high" (~630 tokens avg)
    """
    try:
        UsageTracker.reset()

        if enable_image_annotation:
            api_module.api_image_request = responses_api_image_request
            model_module.api_image_request = responses_api_image_request

        pipeline_options = _build_pipeline_options(
            enable_image_annotation, image_detail
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseV2DocumentBackend,
                )
            }
        )

        result = converter.convert(str(pdf_path))

        full_text = result.document.export_to_markdown()

        if enable_image_annotation:
            full_text, _ = _extract_and_process_images(result.document, full_text)

        doc = result.document
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1

        usage = _calculate_usage(enable_image_annotation)

        return dict(text=full_text, page_count=page_count, usage=usage)

    except Exception as exc:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(exc)}") from exc
