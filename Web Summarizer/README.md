# Web Page Summarizer

A Python-based tool that takes a list of URLs, fetches their content, and generates concise summaries using the OpenAI API. This script efficiently handles multiple URLs, providing streamlined summaries of web page content.

## Features

* **Content Extraction**: Fetches and processes text from each provided URL.
* **Summarization**: Uses GPT-3.5 to generate web page summaries.
* **Rotating User Agents**: Bypasses 403 errors by simulating different browsers.

## Installation

1. Clone the repository.
2. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Replace `YOUR_OPENAI_API_KEY` in the script.

## Usage

1. Run the script:
    
    ```bash
    python url_summarizer.py
    ```
    
2. Paste URLs, one per line, and press Enter twice.

## Dependencies

* `requests`
* `beautifulsoup4`
* `openai`

## Examples

### Example 1: Summarizing a Single URL

```python
# Example URL
url = "https://example.com/article"

# Run the script and provide the URL
python url_summarizer.py
# Paste the URL when prompted

# Expected Output
# URL: https://example.com/article
# Summary:
# - This article discusses the latest trends in technology...
```

### Example 2: Summarizing Multiple URLs

```python
# Example URLs
urls = [
    "https://example.com/article1",
    "https://example.com/article2",
    "https://example.com/article3"
]

# Run the script and provide the URLs
python url_summarizer.py
# Paste each URL on a new line when prompted

# Expected Output
# URL: https://example.com/article1
# Summary:
# - This article covers the impact of AI on healthcare...
# --------------------------------------------------------------------------------
# URL: https://example.com/article2
# Summary:
# - This article explores the advancements in renewable energy...
# --------------------------------------------------------------------------------
# URL: https://example.com/article3
# Summary:
# - This article highlights the importance of cybersecurity in the digital age...
```

## Detailed Explanations

### Content Extraction

The script uses the `requests` library to fetch the content of each URL. It then processes the HTML content using `BeautifulSoup` to extract the text from the `<p>` tags. This ensures that the script captures the main content of the web page.

### Summarization

The extracted text is passed to the OpenAI API, which uses the GPT-3.5 model to generate a concise summary. The script handles the API interaction and formats the summary for easy reading.

### Rotating User Agents

To avoid 403 errors and simulate different browsers, the script rotates through a list of user agents. This helps in fetching content from websites that have restrictions on automated requests.

## Error Handling

The script includes error handling to manage issues such as network errors, invalid URLs, and API errors. It provides informative messages to help users troubleshoot any problems that may arise.

## Future Enhancements

* **Improved Summarization**: Explore advanced summarization techniques to enhance the quality of summaries.
* **Customizable Output**: Allow users to customize the format and length of the summaries.
* **Additional Content Extraction**: Extend content extraction to include other HTML elements such as headings and lists.
* **Batch Processing**: Implement batch processing to handle large lists of URLs more efficiently.

## Additional Examples

### Example 3: Handling Invalid URLs

```python
# Example Invalid URL
url = "https://invalid-url.com"

# Run the script and provide the URL
python url_summarizer.py
# Paste the URL when prompted

# Expected Output
# Error fetching https://invalid-url.com: HTTPError: 404 Client Error: Not Found for url: https://invalid-url.com
```

### Example 4: Customizing Summarization Length

```python
# Modify the summarize_text function in the script to change max_tokens
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=300,  # Increase the number of tokens for a longer summary
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None
```

### Example 5: Extracting Additional Content

```python
# Modify the fetch_content function in the script to extract headings
def fetch_content(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = ' '.join([p.text for p in soup.find_all('p')])
        headings = ' '.join([h.text for h in soup.find_all(['h1', 'h2', 'h3'])])
        return f"{headings}\n{content}".strip()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
```
