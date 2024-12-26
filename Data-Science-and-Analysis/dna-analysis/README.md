# DNA Analysis Tool

A comprehensive DNA sequence analysis tool that provides various functionalities for analyzing genetic data, including basic sequence statistics, composition analysis, pattern matching, and SNP analysis.

## Features

- **Basic Sequence Analysis**
  - Sequence count and statistics
  - File metadata analysis
  - Sample data preview

- **Composition Analysis**
  - Nucleotide frequency counting
  - GC content calculation
  - Sequence validation

- **Pattern Matching**
  - DNA pattern search functionality
  - Position tracking for matches
  - Multiple occurrence handling

- **SNP Analysis**
  - Comprehensive SNP data processing
  - Health insights and trait analysis
  - Genome coverage calculation
  - Interactive visualization dashboard

## Project Structure

```plaintext
dna-analysis/
├── app.py              # Flask web application
├── dna_analyzer.py     # Core DNA analysis functionality
├── dna.ipynb          # Jupyter notebook with SNP analysis
├── file_handler.py    # File I/O operations
└── main.py            # Command-line interface
```

## Usage

1. **Command Line Interface**
```bash
python main.py
```

2. **Web Interface**
```bash
python app.py
```

3. **Jupyter Notebook**
- Open `dna.ipynb` for interactive SNP analysis

## Input Formats

The tool supports various DNA sequence file formats:
- Plain text files with DNA sequences
- SNP data files
- FASTA format
- VCF files

## Features in Detail

### Basic Statistics
- Total sequence count
- File size and metadata
- Column detection
- Sample data preview

### Sequence Composition
- Nucleotide frequency analysis
- GC content calculation
- Sequence validation

### Pattern Matching
- Custom pattern search
- Position tracking
- Occurrence counting

### SNP Analysis
- Health trait analysis
- Wellness insights
- Genome coverage
- Interactive visualizations

## Dependencies

- Python 3.8+
- pandas
- numpy
- Flask
- plotly
- streamlit
- scikit-learn
- cryptography

## Installation

```bash
pip install -r requirements.txt
```

## License

MIT License 