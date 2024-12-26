from typing import List, Dict, Optional
import re
from collections import Counter

class DNAAnalyzer:
    """Analyzes DNA sequence data"""
    
    NUCLEOTIDES = {'A', 'T', 'G', 'C'}
    
    def __init__(self, data: List[str], metadata: Dict):
        self.data = data
        self.metadata = metadata
        self.separator = None
        self.columns = []
        self._detect_structure()
    
    def _detect_structure(self):
        """Detect data structure from first line"""
        if self.data:
            separators = ['\t', ',', ' ']
            first_line = self.data[0]
            for sep in separators:
                if sep in first_line:
                    self.separator = sep
                    self.columns = first_line.split(sep)
                    break
    
    def analyze_sequence_composition(self, sequence_col: Optional[int] = None) -> Dict:
        """Analyze DNA sequence composition"""
        results = {'nucleotides': Counter(), 'gc_content': 0.0}
        
        for line in self.data[1:]:  # Skip header
            if self.separator:
                fields = line.split(self.separator)
                sequence = fields[sequence_col] if sequence_col is not None else fields[-1]
            else:
                sequence = line
            
            # Count nucleotides
            nucleotides = Counter(sequence.upper())
            results['nucleotides'].update(nucleotides)
            
            # Calculate GC content
            gc_count = nucleotides.get('G', 0) + nucleotides.get('C', 0)
            total = len(sequence)
            if total > 0:
                results['gc_content'] = (gc_count / total) * 100
        
        return results
    
    def find_patterns(self, pattern: str) -> List[Dict]:
        """Find specific DNA patterns in sequences"""
        matches = []
        pattern = pattern.upper()
        
        for i, line in enumerate(self.data[1:], 1):  # Skip header
            if self.separator:
                sequence = line.split(self.separator)[-1]
            else:
                sequence = line
                
            sequence = sequence.upper()
            positions = [m.start() for m in re.finditer(pattern, sequence)]
            
            if positions:
                matches.append({
                    'line': i,
                    'positions': positions,
                    'count': len(positions)
                })
        
        return matches
    
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the DNA data"""
        return {
            'total_sequences': len(self.data) - 1,  # Exclude header
            'metadata': self.metadata,
            'columns': self.columns,
            'sample_data': self.data[1:4]  # First 3 sequences
        }
    
    def validate_sequences(self) -> Dict:
        """Validate DNA sequences for errors"""
        results = {
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'errors': []
        }
        
        for i, line in enumerate(self.data[1:], 1):
            if self.separator:
                sequence = line.split(self.separator)[-1]
            else:
                sequence = line
                
            sequence = sequence.upper()
            invalid_chars = set(sequence) - self.NUCLEOTIDES
            
            if invalid_chars:
                results['invalid_sequences'] += 1
                results['errors'].append({
                    'line': i,
                    'invalid_chars': list(invalid_chars)
                })
            else:
                results['valid_sequences'] += 1
        
        return results