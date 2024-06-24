# AI-Powered Research Synthesizer

This project is an AI-powered research assistant that can gather, process, and synthesize information on any given topic, leveraging the Llama 3 70B Instruct model through NVIDIA's API and the LangChain framework.

## Features

- Topic-based information gathering
- Text summarization
- Automatic question generation and answering
- Information synthesis

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your NVIDIA API key in `config.py`

## Usage

Run the main script:

```
python main.py
```

Enter a research topic when prompted. The system will gather information, process it, and provide a synthesis including a summary and relevant questions and answers.

## Project Structure

- `main.py`: Main script to run the application
- `agents/`: Contains Researcher and Synthesizer classes
- `utils/`: Contains utility functions for API calls and text processing
- `config.py`: Configuration file for API keys

## Dependencies

- langchain
- requests
- nltk

## Note

This project uses the NVIDIA API and requires a valid API key to function. Ensure you have the necessary permissions and credits to use the API.