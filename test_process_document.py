"""
Simple script to test the /process-document endpoint
"""

import base64
import requests
import json


def test_process_document():
    # Read the sample PDF
    pdf_path = (
        "data/cookbooks_input/ABADIA-RETUERTA-INFORME-SOSTENIBILIDAD-ESG-2023.pdf"
        # "data/cookbooks_input/sample.pdf"
    )

    with open(pdf_path, "rb") as pdf_file:
        pdf_content = pdf_file.read()
        base64_data = base64.b64encode(pdf_content).decode("utf-8")

    # Prepare the request
    # url = "https://believers-eva-ai-service-production.up.railway.app/process-document"
    url = "http://localhost:8000/process-document"
    payload = {
        "base64_data": base64_data,
        "enable_image_annotation": True,
        "force_ocr": False,
        "lang": "es",
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

        # Save full response to JSON file
        output_file = "response_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Full response saved to: {output_file}")
        print()

        print("Response:")
        print(f"File Type: {result.get('file_type')}")
        print(f"Page Count: {result.get('page_count')}")
        print(f"Text Length: {len(result.get('text', ''))}")

        # Print processing metrics
        if result.get("processing_metrics"):
            metrics = result["processing_metrics"]
            print("\n" + "=" * 50)
            print("PROCESSING METRICS")
            print("=" * 50)
            print(f"Processing Time: {metrics.get('processing_time_seconds')} seconds")
            print(f"Total Chunks: {metrics.get('total_chunks')}")

            if metrics.get("costs"):
                costs = metrics["costs"]
                print("\nCosts Breakdown:")

                # Parse PDF costs
                pdf_costs = costs.get("parse_pdf", {})
                if pdf_costs.get("cost", 0) > 0:
                    print("\n  Parse PDF:")
                    print(f"    Model: {pdf_costs.get('model')}")
                    print(f"    Input Tokens: {pdf_costs.get('input_tokens')}")
                    print(f"    Output Tokens: {pdf_costs.get('output_tokens')}")
                    print(f"    Cost: ${pdf_costs.get('cost'):.4f}")

                # Embeddings costs
                emb_costs = costs.get("embeddings", {})
                if emb_costs.get("cost", 0) > 0:
                    print("\n  Embeddings:")
                    print(f"    Model: {emb_costs.get('model')}")
                    print(f"    Input Tokens: {emb_costs.get('input_tokens')}")
                    print(f"    Cost: ${emb_costs.get('cost'):.4f}")

                # Verifiable data costs
                ver_costs = costs.get("verifiable_data", {})
                if ver_costs.get("cost", 0) > 0:
                    print("\n  Verifiable Data:")
                    print(f"    Model: {ver_costs.get('model')}")
                    print(f"    Input Tokens: {ver_costs.get('input_tokens')}")
                    print(f"    Output Tokens: {ver_costs.get('output_tokens')}")
                    print(f"    Cost: ${ver_costs.get('cost'):.4f}")

                print(f"\n  TOTAL COST: ${costs.get('total_cost'):.4f}")
            print("=" * 50)

        print("\nFirst 500 characters of text:")
        print(result.get("text", "")[:500])

        if result.get("verifiable_data"):
            print(
                f"\nVerifiable Data: {json.dumps(result.get('verifiable_data'), indent=2)}"
            )
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    test_process_document()
