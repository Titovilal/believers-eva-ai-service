"""
Execute Number Detection
Script to demonstrate number detection functionality (digit and text formats).
"""
from base_cookbook import BaseCookbook
from src.rag.detect_number_in_text import detect_number_in_text


class NumberDetectionCookbook(BaseCookbook):
    """Cookbook for number detection functionality."""

    def __init__(self):
        super().__init__("NUMBER DETECTION (DIGIT AND TEXT FORMAT)")

    def run(self):
        self.print_header()

        # Test cases
        test_cases = [
            "I have 3 apples",
            "I have three apples",
            "I have some apples",
            "There are 25 students in the class",
            "There are twenty five students in the class",
            "The meeting is at 3pm",
            "The meeting is at three pm",
            "No numbers here at all",
            "She bought one hundred and fifty books",
            "She bought 150 books",
            "Zero matches found",
            "0 matches found",
            "The price is ten dollars",
            "The price is $10",
            "First place winner",
            "1st place winner",
        ]

        self.print_section("🔍 Testing number detection:")

        for i, text in enumerate(test_cases, 1):
            result = detect_number_in_text(text, lang="en")
            emoji = "✅" if result else "❌"
            status = "HAS NUMBER" if result else "NO NUMBER"
            print(f"\n{i}. {emoji} [{status}]")
            print(f"   Text: \"{text}\"")

        self.print_success("Number detection completed!")


if __name__ == "__main__":
    cookbook = NumberDetectionCookbook()
    cookbook.execute()
