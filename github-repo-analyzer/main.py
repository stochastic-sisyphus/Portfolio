import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import json
import io
import base64
from github import Github
import nltk
from datetime import datetime, timedelta
import functools

nltk.download('punkt', quiet=True)

g = Github(os.environ.get('GITHUB_TOKEN'))

cache = {}

def cache_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cache:
            result, timestamp = cache[key]
            if datetime.now() - timestamp < timedelta(hours=1):
                return result
        result = func(*args, **kwargs)
        cache[key] = (result, datetime.now())
        return result
    return wrapper

@cache_result
def get_starred_repos(username):
    user = g.get_user(username)
    return list(user.get_starred())

def get_readme_content(repo):
    try:
        readme = repo.get_readme()
        return readme.decoded_content.decode('utf-8')
    except:
        return ""

def simple_summarize(text, sentences_count=3):
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences[:sentences_count])

@cache_result
def process_repos(repos):
    data = []
    for repo in repos:
        readme_content = get_readme_content(repo)
        summary = simple_summarize(readme_content)
        data.append({
            'name': repo.name,
            'description': repo.description,
            'language': repo.language,
            'stars': repo.stargazers_count,
            'forks': repo.forks_count,
            'url': repo.html_url,
            'topics': repo.get_topics(),
            'created_at': repo.created_at,
            'summary': summary
        })
    return pd.DataFrame(data)

def visualize_languages(df):
    lang_counts = df['language'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%')
    ax.set_title('Language Distribution')
    return fig

def visualize_stars_over_time(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')
    df['cumulative_stars'] = df['stars'].cumsum()
    fig, ax = plt.subplots()
    ax.plot(df['created_at'], df['cumulative_stars'])
    ax.set_title('Cumulative Stars Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Stars')
    return fig

def analyze_topics(df):
    all_topics = [topic for topics in df['topics'] for topic in topics]
    topic_counts = pd.Series(all_topics).value_counts()
    return topic_counts.head(10)

def suggest_interesting_repos(df):
    df['score'] = df['stars'] + df['forks'] * 2 + (pd.Timestamp.now() - df['created_at']).dt.days * -0.1
    return df.sort_values('score', ascending=False).head(5)

def generate_markdown_summary(df):
    total_repos = len(df)
    markdown = f"{total_repos} starred repositories\n\n"
    for _, repo in df.iterrows():
        markdown += f"- [{repo['name']}]({repo['url']})\n"
        markdown += f"  - Description: {repo['description']}\n"
        markdown += f"  - Key points: {repo['summary']}\n\n"
    return markdown

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def serve_file(self, filename, content_type):
        with open(filename, 'rb') as file:
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(file.read())

    def do_GET(self):
        if self.path == '/styles.css':
            self.serve_file('styles.css', 'text/css')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
            <html>
            <head>
                <link rel="stylesheet" type="text/css" href="/styles.css">
            </head>
            <body>
                <form method="post">
                    <input type="text" name="username" placeholder="Enter GitHub username">
                    <input type="submit" value="Analyze">
                </form>
            </body>
            </html>
            ''')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        username = parse_qs(post_data)['username'][0]

        repos = get_starred_repos(username)
        df = process_repos(repos)

        top_repos = df[['name', 'stars', 'language']].sort_values('stars', ascending=False).head().to_html()
        
        lang_dist = visualize_languages(df)
        img_lang = io.BytesIO()
        lang_dist.savefig(img_lang, format='png')
        img_lang.seek(0)
        img_lang_url = base64.b64encode(img_lang.getvalue()).decode()

        stars_time = visualize_stars_over_time(df)
        img_stars = io.BytesIO()
        stars_time.savefig(img_stars, format='png')
        img_stars.seek(0)
        img_stars_url = base64.b64encode(img_stars.getvalue()).decode()

        top_topics = analyze_topics(df).to_frame().to_html()
        
        interesting = suggest_interesting_repos(df)
        interesting_repos = "<br>".join([
            f"<b>{repo['name']}</b><br>"
            f"Description: {repo['description']}<br>"
            f"Summary: {repo['summary']}<br>"
            f"Stars: {repo['stars']}, Forks: {repo['forks']}<br>"
            f"URL: <a href='{repo['url']}'>{repo['url']}</a><br>"
            for _, repo in interesting.iterrows()
        ])
        
        markdown_summary = generate_markdown_summary(df)

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(f'''
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="/styles.css">
        </head>
        <body>
            <h1>GitHub Repo Analyzer Results for {username}</h1>
            <h2>Top 5 starred repositories</h2>
            {top_repos}
            <h2>Language Distribution</h2>
            <img src="data:image/png;base64,{img_lang_url}">
            <h2>Cumulative Stars Over Time</h2>
            <img src="data:image/png;base64,{img_stars_url}">
            <h2>Top 10 Topics</h2>
            {top_topics}
            <h2>Suggested interesting repositories</h2>
            {interesting_repos}
            <h2>Markdown Summary</h2>
            <textarea rows="10" cols="50">{markdown_summary}</textarea>
        </body>
        </html>
        '''.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
