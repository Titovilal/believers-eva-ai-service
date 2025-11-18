"""
Base Cookbook Class
Provides common functionality for all cookbook scripts to avoid code duplication.
"""

import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Configure environment when module is imported
load_dotenv()
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class BaseCookbook:
    """Base class for cookbook execution scripts."""

    def __init__(self, title: str):
        """
        Initialize the cookbook.

        Args:
            title: The title to display in the header
        """
        self.title = title

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get the data/cookbooks_input directory."""
        return self.project_root / "data" / "cookbooks_input"

    @property
    def output_dir(self) -> Path:
        """Get or create the outputs directory."""
        output_dir = self.project_root / "data" / "cookbooks_output"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def print_header(self, subtitle: Optional[str] = None):
        """
        Print a formatted header.

        Args:
            subtitle: Optional subtitle to display below the title
        """
        print("=" * 80)
        print(self.title)
        print("=" * 80)
        if subtitle:
            print(f"\n{subtitle}")
            print("-" * 80)

    def print_section(self, text: str):
        """Print a section separator with text."""
        print(f"\n{text}")
        print("-" * 80)

    def print_success(self, message: str, file_path: Optional[Path] = None):
        """
        Print a success message.

        Args:
            message: Success message to display
            file_path: Optional file path to display
        """
        print("\n" + "=" * 80)
        print(f"✅ {message}")
        if file_path:
            print(f"📁 Saved to: {file_path}")
        print("=" * 80)

    def print_error(self, error: Exception):
        """
        Print an error message with traceback.

        Args:
            error: The exception to display
        """
        print(f"\n❌ Error: {str(error)}")
        import traceback

        traceback.print_exc()

    def read_file(self, file_path: Path) -> str:
        """
        Read a text file.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def save_text_file(self, content: str, filename: str) -> Path:
        """
        Save content to a text file in the output directory.

        Args:
            content: Content to save
            filename: Name of the output file

        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def save_json_file(self, data: dict, filename: str) -> Path:
        """
        Save data to a JSON file in the output directory.

        Args:
            data: Dictionary to save as JSON
            filename: Name of the output file

        Returns:
            Path to the saved file
        """
        import json

        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

    def run(self):
        """
        Main execution method to be implemented by subclasses.
        Should contain the cookbook-specific logic.
        """
        raise NotImplementedError("Subclasses must implement the run() method")

    def execute(self):
        """Execute the cookbook with error handling."""
        try:
            self.run()
        except Exception as e:
            self.print_error(e)
