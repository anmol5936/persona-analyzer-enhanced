import re
import nltk
from nltk.tokenize import sent_tokenize

class ContentRefiner:
    """
    Cleans, refines, and summarizes text content from document sections.
    """

    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def refine_text(self, text: str) -> str:
        """
        Cleans and summarizes the text to be concise and relevant.

        Args:
            text: The raw text content from a document section.

        Returns:
            A cleaned and summarized version of the text.
        """
        # Basic cleaning: remove extra whitespace and non-standard characters
        cleaned_text = self._basic_clean(text)

        # Summarize to a target length
        summarized_text = self._summarize(cleaned_text)

        return summarized_text

    def _basic_clean(self, text: str) -> str:
        """Performs basic text cleaning."""
        # Replace multiple newlines/spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove any characters that are not alphanumeric, punctuation, or spaces
        text = re.sub(r'[^\w\s.,-?()\'"]', '', text)
        return text.strip()

    def _summarize(self, text: str) -> str:
        """
        Summarizes the text to be under the max_length, preserving key sentences.
        """
        # If text is already short enough, return it
        if len(text) <= self.max_length:
            return text

        # Tokenize into sentences
        sentences = sent_tokenize(text)

        # Simple heuristic: keep the first few sentences as they often contain key info
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) + 1 <= self.max_length:
                summary += sentence + " "
            else:
                break

        # If the first sentence alone is too long, truncate it
        if not summary and sentences:
            summary = sentences[0][:self.max_length-3] + "..."

        return summary.strip()

# Example Usage
if __name__ == '__main__':
    refiner = ContentRefiner(max_length=150)

    long_text = (
        "The South of France, also known as the French Riviera, is a stunning region "
        "with a rich history and vibrant culture. It's famous for its beautiful beaches, "
        "picturesque villages, and delicious cuisine. This guide will explore the top "
        "destinations, including Nice, Cannes, and Saint-Tropez. We will provide tips "
        "for accommodation, dining, and activities to help you plan the perfect trip. "
        "Whether you are a history buff, a foodie, or a sun-seeker, the South of "
        "France has something to offer for everyone."
    )

    refined_content = refiner.refine_text(long_text)
    print("Original Length:", len(long_text))
    print("Refined Length:", len(refined_content))
    print("Refined Content:", refined_content)

    short_text = "A short and sweet description."
    refined_short = refiner.refine_text(short_text)
    print("\nOriginal Short Text:", short_text)
    print("Refined Short Text:", refined_short)
