Web Page Summarizer
A Python-based tool that takes a list of URLs, fetches their content, and generates concise summaries using the OpenAI API. This script is designed to handle multiple URLs at once, providing an efficient way to summarize content from various web pages.

Features
Content Extraction: Fetches and processes text from each provided URL.
Summarization: Uses GPT-3.5 to generate summaries of web page content.
Rotating User Agents: Bypasses common 403 errors by simulating different browsers.
Installation
Clone the repository.
Install dependencies:
pip install -r requirements.txt
Replace YOUR_OPENAI_API_KEY in the script with your OpenAI API key.
Usage
Run the script:
python url_summarizer.py
Paste URLs, one per line, and press Enter twice to generate summaries.
Dependencies
requests
beautifulsoup4
openai
