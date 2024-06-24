from utils.nvidia_api import web_search, extract_information

class Researcher:
    def gather_data(self, topic):
        # Use NVIDIA's function to get general information (simulating web search)
        search_results = web_search(topic)
        
        # Extract relevant information from the search results
        extracted_info = extract_information(search_results, topic)
        
        return extracted_info
