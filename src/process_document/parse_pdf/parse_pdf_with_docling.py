"""
PDF Parser using Docling library
Uses Docling for document conversion with optional image annotation.
"""

import gc
import re
import base64
import threading
import uuid
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
from ...utils.constants import PARSE_PDF, calculate_cost

DOCLING_CONFIG = PARSE_PDF["docling"]
DOCLING_SYSTEM_PROMPT = "Describe the picture."

# Global dict of usage lists keyed by request_id (thread-safe)
_usage_records = {}
_records_lock = threading.Lock()


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

    # Extract request_id (internal use only)
    request_id = params.pop("_request_id", None)

    # Build payload without request_id
    payload = _build_responses_request_payload(prompt, image_base64, params)

    try:
        response = client.responses.create(**payload)
        generated_text = response.output_text.strip()
        num_tokens = response.usage.input_tokens + response.usage.output_tokens
        stop_reason = VlmStopReason.END_OF_SEQUENCE

        # Record usage in the list for this request_id
        if request_id and request_id in _usage_records:
            with _records_lock:
                _usage_records[request_id].append(
                    {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                )

        return generated_text, num_tokens, stop_reason

    except Exception as exc:
        log_exception("Error annotating image description", exc)
        raise


def _build_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
    cost: float = 0,
    mode: str = "docling",
) -> dict:
    return dict(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cost=cost,
        mode=mode,
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
    enable_image_annotation: bool, image_detail: str, request_id: str
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
            _request_id=request_id,  # Pass request_id through params
        ),
        prompt=DOCLING_SYSTEM_PROMPT,
        concurrency=DOCLING_CONFIG["concurrency"],
    )

    return pipeline_options


def _calculate_usage(enable_image_annotation: bool, request_id: str) -> dict:
    if not enable_image_annotation:
        return _build_usage()

    # Sum all usage records for this request_id
    records = _usage_records.get(request_id, [])
    if not records:
        return _build_usage()

    input_tokens = sum(r["input_tokens"] for r in records)
    output_tokens = sum(r["output_tokens"] for r in records)

    cost = calculate_cost(DOCLING_CONFIG["model_id"], input_tokens, output_tokens)

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
    # Generate unique request ID
    request_id = str(uuid.uuid4())

    converter = None
    result = None
    doc = None

    try:
        # Initialize empty list for this request
        with _records_lock:
            _usage_records[request_id] = []

        if enable_image_annotation:
            api_module.api_image_request = responses_api_image_request
            model_module.api_image_request = responses_api_image_request

        pipeline_options = _build_pipeline_options(
            enable_image_annotation, image_detail, request_id
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

        usage = _calculate_usage(enable_image_annotation, request_id)

        return dict(text=full_text, page_count=page_count, usage=usage)

    except Exception as exc:
        raise Exception(f"Error parsing PDF {pdf_path}: {str(exc)}") from exc
    finally:
        # Cleanup records
        with _records_lock:
            _usage_records.pop(request_id, None)

        # Explicitly cleanup large objects to free memory
        del doc
        del result
        del converter

        # Force garbage collection
        gc.collect()
