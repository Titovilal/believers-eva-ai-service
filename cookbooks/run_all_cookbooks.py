#!/usr/bin/env python3
"""
Run all cookbook scripts in parallel with live progress tracking.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm


@dataclass
class CookbookResult:
    """Result of running a cookbook."""

    name: str
    success: bool
    execution_time: float = 0.0
    error_output: Optional[str] = None


def run_cookbook(script_name: str, pbar: tqdm) -> CookbookResult:
    """Run a single cookbook script and capture its result."""
    pbar.set_description(f"⏳ {script_name}")
    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        execution_time = time.time() - start_time
        pbar.set_description(f"✓ {script_name}")
        pbar.update(1)
        return CookbookResult(
            name=script_name, success=True, execution_time=execution_time
        )

    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        error_msg = e.stderr.strip() if e.stderr else e.stdout.strip()
        pbar.set_description(f"✗ {script_name}")
        pbar.update(1)
        return CookbookResult(
            name=script_name,
            success=False,
            execution_time=execution_time,
            error_output=error_msg,
        )

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        pbar.set_description(f"✗ {script_name} (timeout)")
        pbar.update(1)
        return CookbookResult(
            name=script_name,
            success=False,
            execution_time=execution_time,
            error_output="Timeout after 5 minutes",
        )

    except Exception as e:
        execution_time = time.time() - start_time
        pbar.set_description(f"✗ {script_name}")
        pbar.update(1)
        return CookbookResult(
            name=script_name,
            success=False,
            execution_time=execution_time,
            error_output=str(e),
        )


def print_summary(results: list[CookbookResult]) -> None:
    """Print a summary of all cookbook executions."""
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for result in successful:
        print(f"  • {result.name} ({result.execution_time:.2f}s)")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for result in failed:
            print(f"  • {result.name}")
            if result.error_output:
                # Show first 200 chars of error
                error_preview = result.error_output[:200]
                if len(result.error_output) > 200:
                    error_preview += "..."
                print(f"    Error: {error_preview}")

    print("\n" + "=" * 60)


def main():
    """Execute all cookbooks in parallel."""
    parser = argparse.ArgumentParser(
        description="Run all cookbook scripts in parallel with live progress tracking."
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=2,
        help="Number of cookbooks to run concurrently (default: 2)",
    )
    args = parser.parse_args()

    cookbooks = [
        "execute_parse_pdf.py",
        "execute_detect_number_in_text.py",
        "execute_generate_chunks.py",
        "execute_generate_embeddings.py",
    ]

    max_workers = args.workers
    print(f"\nRunning {len(cookbooks)} cookbooks (concurrency: {max_workers})")
    print("💡 Tip: Use -w or --workers to change concurrency (e.g., -w 4)\n")

    # Execute in parallel and collect results
    results = []
    with tqdm(
        total=len(cookbooks), bar_format="{desc} | {n_fmt}/{total_fmt} completed"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_cookbook, cb, pbar): cb for cb in cookbooks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    # Print summary
    print_summary(results)

    # Exit with error code if any failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
