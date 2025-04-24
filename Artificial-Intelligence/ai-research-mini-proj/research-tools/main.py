from agents.synthesizer import Synthesizer
from agents.researcher import Researcher
from utils.text_processor import process_text

def main():
    synthesizer = Synthesizer()
    researcher = Researcher()

    while True:
        try:
            topic = input("Enter a research topic (or 'quit' to exit): ")
            if topic.lower() == 'quit':
                break

            print("Gathering data...")
            raw_data = researcher.gather_data(topic)
            
            print("Processing data...")
            processed_data = process_text(raw_data)
            
            print("Synthesizing information...")
            synthesis = synthesizer.synthesize(topic, processed_data)

            print(f"\nSynthesis on {topic}:")
            print(synthesis)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different topic.")

if __name__ == "__main__":
    main()