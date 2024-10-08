import requests
from bs4 import BeautifulSoup
import openai
import random

# Set your OpenAI API key
openai.api_key = 'OPENAI_KEY'  # Replace with your OpenAI API key

# List of user agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
]

# Function to fetch the content of a webpage with headers to avoid 403 errors
def fetch_content(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),  # Rotate user agents
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get text from the body of the page
        content = ' '.join([p.text for p in soup.find_all('p')])
        return content.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to summarize text using OpenAI with the latest model
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the latest model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

# Function to summarize a list of URLs
def summarize_urls(urls):
    summaries = []
    for url in urls:
        print(f"\nFetching content from: {url}")
        content = fetch_content(url)
        if content:
            print("Generating summary...")
            summary = summarize_text(content)
            if summary:
                summaries.append({
                    'url': url,
                    'summary': summary
                })
    return summaries

# Function to display and save summaries in a user-friendly format
def display_summaries(summaries):
    print("\nSummarized Content:")
    with open("summarized_content.txt", "w") as file:
        for summary in summaries:
            formatted_summary = f"URL: {summary['url']}\nSummary:\n- {summary['summary']}\n{'-'*80}\n"
            print(formatted_summary)
            file.write(formatted_summary)
    print("\nSummaries saved to summarized_content.txt")

# Main function to run the summarizer
def main():
    print("Paste the URLs you want to summarize, one per line (press Enter twice to finish):")
    urls_input = []
    while True:
        line = input()
        if line.strip() == "":
            break
        urls_input.append(line.strip())
    urls = [url for url in urls_input if url]  # Remove any empty strings
    if not urls:
        print("No URLs provided.")
        return
    summaries = summarize_urls(urls)
    if summaries:
        display_summaries(summaries)

if __name__ == "__main__":
    main()
