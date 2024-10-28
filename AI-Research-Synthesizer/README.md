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

### Example Output

Here is an example of the output you can expect:

```
Enter a research topic (or 'quit' to exit): Artificial Intelligence
Gathering data...
Processing data...
Synthesizing information...

Synthesis on Artificial Intelligence:

Summary: Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. It encompasses various subfields such as machine learning, natural language processing, and robotics.

Q1: What is Artificial Intelligence?
A1: Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior.

Q2: What are the subfields of AI?
A2: The subfields of AI include machine learning, natural language processing, and robotics.

Q3: How is AI used in robotics?
A3: AI is used in robotics to enable machines to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.
```

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


## To view example outputs please see the photo pdf in this folder. 
[View the PDF](https://github.com/stochastic-sisyphus/Portfolio/blob/main/AI-Research-Synthesizer/Photo.pdf)
