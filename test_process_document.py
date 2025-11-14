"""
Simple script to test the /process-document endpoint
"""

import base64
import requests
import json


def test_process_document():
    # Read the sample PDF
    pdf_path = "data/cookbooks_input/sample.pdf"

    with open(pdf_path, "rb") as pdf_file:
        pdf_content = pdf_file.read()
        base64_data = base64.b64encode(pdf_content).decode("utf-8")

    # Prepare the request
    url = "https://believers-eva-ai-service-production.up.railway.app/process-document"
    payload = {
        "base64_data": base64_data,
        "enable_image_annotation": True,
        "force_ocr": False,
        "lang": "en",
    }

    print(f"Testing endpoint: {url}")
    print(f"Processing file: {pdf_path}")
    print("-" * 50)

    # Make the request
    response = requests.post(url, json=payload)

    # Print the response
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

    if response.status_code == 200:
        result = response.json()
        print("Response:")
        print(f"File Type: {result.get('file_type')}")
        print(f"Page Count: {result.get('page_count')}")
        print(f"Chunk Count: {result.get('chunk_count')}")
        print(f"Text Length: {len(result.get('text', ''))}")
        print(f"Number of Embeddings: {len(result.get('embeddings', []))}")
        print("\nFirst 500 characters of text:")
        print(result.get("text", "")[:500])

        if result.get("verifiable_facts"):
            print(
                f"\nVerifiable Facts: {json.dumps(result.get('verifiable_facts'), indent=2)}"
            )
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    test_process_document()
