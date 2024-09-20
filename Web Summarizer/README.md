**Web Page Summarizer**

A Python-based tool that takes a list of URLs, fetches their content, and generates concise summaries using the OpenAI API. This script efficiently handles multiple URLs, providing streamlined summaries of web page content.

### Features

*   **Content Extraction**: Fetches and processes text from each provided URL.
*   **Summarization**: Uses GPT-3.5 to generate web page summaries.
*   **Rotating User Agents**: Bypasses 403 errors by simulating different browsers.

### Installation

1.  Clone the repository.
2.  Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3.  Replace `YOUR_OPENAI_API_KEY` in the script.

### Usage

1.  Run the script:
    
    ```bash
    python url_summarizer.py
    ```
    
2.  Paste URLs, one per line, and press Enter twice.

### Dependencies

*   `requests`
*   `beautifulsoup4`
*   `openai`
