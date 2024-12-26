import os
import platform
from typing import Optional
from file_handler import FileHandler
from dna_analyzer import DNAAnalyzer

def get_dna_file() -> Optional[str]:
    """Prompt user for DNA file path and validate it"""
    while True:
        try:
            file_path = input("\nEnter the path to your DNA file (or 'q' to quit): ").strip()
            
            if file_path.lower() == 'q':
                return None
            
            is_valid, message = FileHandler.validate_file(file_path)
            if not is_valid:
                print(f"Error: {message}")
                continue
                
            return file_path
            
        except KeyboardInterrupt:
            return None
        except Exception as e:
            print(f"Error: {str(e)}")

def display_menu() -> str:
    """Display analysis options menu"""
    print("\nDNA Analysis Options:")
    print("1. Basic Statistics")
    print("2. Sequence Composition Analysis")
    print("3. Find DNA Patterns")
    print("4. Validate Sequences")
    print("5. Exit")
    return input("Select an option (1-5): ").strip()

def perform_analysis(analyzer: DNAAnalyzer, option: str):
    """Perform selected analysis"""
    if option == '1':
        stats = analyzer.get_basic_stats()
        print("\nBasic Statistics:")
        print(f"Total sequences: {stats['total_sequences']}")
        print(f"File size: {stats['metadata']['file_size'] / 1024:.2f} KB")
        print(f"Columns detected: {stats['columns']}")
        print("\nSample Data:")
        for line in stats['sample_data']:
            print(line)
            
    elif option == '2':
        composition = analyzer.analyze_sequence_composition()
        print("\nSequence Composition:")
        for nucleotide, count in composition['nucleotides'].items():
            print(f"{nucleotide}: {count}")
        print(f"GC Content: {composition['gc_content']:.2f}%")
        
    elif option == '3':
        pattern = input("Enter DNA pattern to search (e.g., ATCG): ").strip().upper()
        matches = analyzer.find_patterns(pattern)
        print(f"\nFound {len(matches)} sequences containing '{pattern}'")
        for match in matches[:5]:  # Show first 5 matches
            print(f"Line {match['line']}: {match['count']} occurrences at positions {match['positions']}")
            
    elif option == '4':
        validation = analyzer.validate_sequences()
        print("\nSequence Validation Results:")
        print(f"Valid sequences: {validation['valid_sequences']}")
        print(f"Invalid sequences: {validation['invalid_sequences']}")
        if validation['errors']:
            print("\nFirst 5 errors:")
            for error in validation['errors'][:5]:
                print(f"Line {error['line']}: Invalid characters {error['invalid_chars']}")

def main():
    """Main program loop"""
    print("DNA File Analyzer")
    print("=" * 30)
    print(f"Python Version: {platform.python_version()}")
    
    try:
        file_path = get_dna_file()
        if not file_path:
            print("\nAnalysis cancelled.")
            return
            
        # Read and process file
        data, metadata = FileHandler.read_file(file_path)
        analyzer = DNAAnalyzer(data, metadata)
        
        # Analysis loop
        while True:
            option = display_menu()
            
            if option == '5':
                print("\nExiting analysis.")
                break
                
            if option not in {'1', '2', '3', '4'}:
                print("Invalid option. Please select 1-5.")
                continue
                
            perform_analysis(analyzer, option)
            
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    main()