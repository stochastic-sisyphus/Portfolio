# GitHub Repo Analyzer

GitHub Repo Analyzer is a web application that analyzes a user's starred GitHub repositories, providing insights and summaries.

## Live Demo

You can try out the live version of this application here:
[GitHub Repo Analyzer on Replit](https://silky-ripe-autoresponder-stochastic-sisyphus.replit.app)

## Features

- Fetch and analyze starred repositories for a given GitHub username
- Generate summaries of repository README files
- Visualize language distribution of starred repos
- Show cumulative stars over time
- Analyze and display top topics
- Suggest interesting repositories based on a scoring system
- Generate a downloadable markdown summary of all repos

## Technologies Used

- Python 3.10+
- Flask web framework
- PyGithub for GitHub API interactions
- Pandas for data manipulation
- Matplotlib for data visualization
- NLTK and sumy for text summarization

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/stochastic-sisyphus/github-repo-analyzer.git
   cd github-repo-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your GitHub token:
   - Create a personal access token on GitHub
   - Set it as an environment variable:
     ```
     export GITHUB_TOKEN=your_token_here
     ```

4. Run the application:
   ```
   python main.py
   ```

5. Open a web browser and navigate to `http://localhost:8080`

## Usage

1. Enter a GitHub username in the provided form
2. Click "Analyze" to process the user's starred repositories
3. View the generated insights, including:
   - Top 5 starred repositories
   - Language distribution
   - Cumulative stars over time
   - Top 10 topics
   - Suggested interesting repositories
4. Download the markdown summary if desired

## Deployment

This application is deployed on Replit. To deploy your own instance:

1. Fork this repository to your Replit account
2. Set up the `GITHUB_TOKEN` secret in your Replit project
3. Click the "Run" button in Replit

## Recent Updates

- Updated the caching mechanism to improve performance
- Enhanced the text summarization algorithm for better accuracy
- Improved the visualization of cumulative stars over time

## Additional Information

- The application now includes a caching mechanism to reduce the number of API calls and improve performance.
- The text summarization algorithm has been enhanced to provide more accurate summaries of repository README files.
- The visualization of cumulative stars over time has been improved for better clarity and understanding.
