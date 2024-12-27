# Core imports
import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
import base64
import pickle
from io import StringIO

# Data processing
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

# Network
import requests

# PDF generation
from fpdf import FPDF, XPos, YPos

# Constants
BASE_URL_CLINVAR = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_URL_DBSNP = "https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/"
BASE_URL_PHARMGKB = "https://api.pharmgkb.org/v1/data/clinicalAnnotation"
CACHE_FILE = "dna_analysis_cache.pkl"
CHUNK_SIZE = 10000
VALID_CHROMOSOMES = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                    '21', '22', 'X', 'Y', 'MT'}
RSID_PATTERN = re.compile(r"^(rs|i)\d+$")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GenomeDataValidator:
    """Enhanced genome data validator with comprehensive checks"""
    
    def __init__(self):
        self.valid_chromosomes = VALID_CHROMOSOMES
        # Expand valid bases to include D and I for insertions/deletions
        self.valid_bases = {'A', 'C', 'G', 'T', 'D', 'I', '-'}
        self.rsid_pattern = RSID_PATTERN
    
    def validate_file(self, file_path: str, chunk_size: int = CHUNK_SIZE) -> Tuple[bool, str]:
        """
        Validate genome file with comprehensive checks using chunked reading
        Returns: (is_valid: bool, error_message: str)
        """
        try:
            # Check file existence and readability
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            if not os.access(file_path, os.R_OK):
                return False, f"File not readable: {file_path}"
            
            # Validate file format using chunks
            total_rows = 0
            invalid_rows = []
            
            for chunk_num, chunk in enumerate(pd.read_csv(file_path, sep='\t', 
                                                         names=['rsid', 'chromosome', 'position', 'genotype'],
                                                         comment='#',
                                                         chunksize=chunk_size)):
                
                # Validate column presence
                if not all(col in chunk.columns for col in ['rsid', 'chromosome', 'position', 'genotype']):
                    return False, "Missing required columns"
                
                # Validate each row in the chunk
                for idx, row in chunk.iterrows():
                    total_rows += 1
                    row_num = chunk_num * chunk_size + idx + 1
                    
                    # Validate rsid format
                    if not self.rsid_pattern.match(str(row['rsid'])):
                        invalid_rows.append((row_num, f"Invalid rsid format: {row['rsid']}"))
                        continue
                    
                    # Validate chromosome
                    if str(row['chromosome']) not in self.valid_chromosomes:
                        invalid_rows.append((row_num, f"Invalid chromosome: {row['chromosome']}"))
                        continue
                    
                    # Validate position
                    try:
                        pos = int(row['position'])
                        if pos <= 0:
                            invalid_rows.append((row_num, f"Invalid position: {pos}"))
                            continue
                    except (ValueError, TypeError):
                        invalid_rows.append((row_num, f"Invalid position format: {row['position']}"))
                        continue
                    
                    # Validate genotype
                    if not self._validate_genotype(str(row['genotype'])):
                        invalid_rows.append((row_num, f"Invalid genotype: {row['genotype']}"))
                        continue
                
                # Log progress for large files
                if (chunk_num + 1) % 10 == 0:
                    logging.info(f"Validated {total_rows:,} rows...")
            
            # Generate validation report
            if invalid_rows:
                error_report = "\n".join([f"Row {row}: {error}" for row, error in invalid_rows[:10]])
                if len(invalid_rows) > 10:
                    error_report += f"\n... and {len(invalid_rows) - 10} more errors"
                return False, f"Validation failed:\n{error_report}"
            
            logging.info(f"Successfully validated {total_rows:,} rows")
            return True, "Validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_genotype(self, genotype: str) -> bool:
        """Validate genotype format and content"""
        # Accept -- for no-calls
        if genotype == '--':
            return True
        
        # Accept D, I, DD, II, and DI for structural variants
        if genotype in {'D', 'I', 'DD', 'II', 'DI'}:
            return True
        
        # For standard nucleotides, check length and valid bases
        if len(genotype) not in [1, 2]:
            return False
        
        return all(base in {'A', 'C', 'G', 'T'} for base in genotype)

class APIClient:
    """Handles API requests with caching and rate limiting"""
    
    def __init__(self):
        self.session = requests.Session()
        self.cache = {}
        self.last_request_time = {}
        self.min_request_interval = 0.5  # seconds between requests
    
    def fetch_snp_data(self, rsid: str) -> Optional[Dict]:
        """Fetch SNP data from multiple sources with caching"""
        if rsid in self.cache:
            return self.cache[rsid]
        
        data = {}
        try:
            # ClinVar data
            clinvar_data = self._make_request(
                BASE_URL_CLINVAR,
                params={'db': 'clinvar', 'term': rsid, 'retmode': 'json'}
            )
            if clinvar_data:
                data['clinvar'] = clinvar_data
            
            # dbSNP data
            dbsnp_url = f"{BASE_URL_DBSNP}{rsid.replace('rs', '')}"
            dbsnp_data = self._make_request(dbsnp_url)
            if dbsnp_data:
                data['dbsnp'] = dbsnp_data
            
            # PharmGKB data
            pharmgkb_data = self._make_request(
                BASE_URL_PHARMGKB,
                params={'rsid': rsid}
            )
            if pharmgkb_data:
                data['pharmgkb'] = pharmgkb_data
            
            self.cache[rsid] = data
            return data
            
        except Exception as e:
            logging.warning(f"Error fetching data for {rsid}: {str(e)}")
            return None
    
    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make API request with rate limiting and retries"""
        for attempt in range(max_retries):
            try:
                # Implement rate limiting
                self._respect_rate_limit(url)
                
                response = self.session.get(url, params=params, timeout=10)
                self.last_request_time[url] = time.time()
                
                if response.status_code == 429:  # Too Many Requests
                    wait_time = float(response.headers.get('Retry-After', 60))
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"API request failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _respect_rate_limit(self, url: str):
        """Ensure minimum time between requests to same endpoint"""
        if url in self.last_request_time:
            elapsed = time.time() - self.last_request_time[url]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

class UnifiedDNAAnalyzer:
    """Main DNA analysis class with enhanced features"""
    
    # Add trait definitions as a class variable
    TRAIT_DEFINITIONS = {
        'Fun & Unique Traits': {
            'Photic Sneeze Reflex': {
                'rs10427255': {'risk_allele': 'C', 'weight': 1.0, 
                              'description': 'ACHOO syndrome - tendency to sneeze when exposed to bright light'},
            },
            'Cilantro Taste Perception': {
                'rs72921001': {'risk_allele': 'A', 'weight': 1.0,
                              'description': 'Determines if cilantro tastes like soap to you'},
                'rs72921002': {'risk_allele': 'T', 'weight': 0.8,
                              'description': 'Secondary marker for cilantro taste perception'}
            },
            'Perfect Pitch': {
                'rs9854612': {'risk_allele': 'T', 'weight': 1.0,
                              'description': 'Associated with ability to identify musical notes without reference'},
            },
            'Circadian Rhythm': {
                'rs12946049': {'risk_allele': 'T', 'weight': 1.0,
                              'description': 'Influences whether you\'re naturally a morning or evening person'},
                'rs2075929': {'risk_allele': 'G', 'weight': 0.8,
                             'description': 'CLOCK gene variant affecting sleep patterns'},
                'rs1801260': {'risk_allele': 'C', 'weight': 0.9,
                             'description': 'Affects sleep timing preferences'}
            },
            'Pain Sensitivity': {
                'rs6746030': {'risk_allele': 'A', 'weight': 1.0,
                             'description': 'SCN9A gene - affects pain threshold'},
                'rs1799971': {'risk_allele': 'G', 'weight': 0.9,
                             'description': 'OPRM1 gene - influences pain perception'}
            }
        },
        'Environmental Response': {
            'UV Sensitivity': {
                'rs1805007': {'risk_allele': 'T', 'weight': 1.0,
                             'description': 'MC1R gene - affects skin response to sunlight'},
                'rs1805008': {'risk_allele': 'T', 'weight': 0.9,
                             'description': 'Influences sunburn susceptibility'}
            },
            'Temperature Adaptation': {
                'rs10509954': {'risk_allele': 'A', 'weight': 1.0,
                              'description': 'UCP1 gene - affects cold tolerance'},
                'rs1799983': {'risk_allele': 'T', 'weight': 0.8,
                             'description': 'NOS3 gene - influences heat adaptation'}
            },
            'Altitude Response': {
                'rs6756667': {'risk_allele': 'G', 'weight': 1.0,
                             'description': 'EPAS1 gene - affects adaptation to high altitudes'},
                'rs1799945': {'risk_allele': 'G', 'weight': 0.9,
                             'description': 'HFE gene - influences oxygen transport'}
            }
        },
        'Nutrition & Diet': {
            'Caffeine Metabolism': {
                'rs762551': {'risk_allele': 'C', 'weight': 1.0,
                            'description': 'CYP1A2 - determines caffeine processing speed'},
                'rs2472297': {'risk_allele': 'T', 'weight': 0.8,
                             'description': 'Influences caffeine consumption habits'},
                'rs4410790': {'risk_allele': 'C', 'weight': 0.7,
                             'description': 'AHR gene - affects caffeine sensitivity'}
            },
            'Carbohydrate Response': {
                'rs7903146': {'risk_allele': 'T', 'weight': 1.0,
                             'description': 'TCF7L2 gene - affects glucose metabolism'},
                'rs1801282': {'risk_allele': 'G', 'weight': 0.9,
                             'description': 'PPARG - influences insulin sensitivity'}
            },
            'Fat Metabolism': {
                'rs1799883': {'risk_allele': 'T', 'weight': 1.0,
                             'description': 'FABP2 gene - affects dietary fat processing'},
                'rs5082': {'risk_allele': 'C', 'weight': 0.8,
                          'description': 'APOA2 - influences fat intake response'}
            }
        },
        'Exercise Response': {
            'Muscle Type': {
                'rs1815739': {'risk_allele': 'T', 'weight': 1.0,
                             'description': 'ACTN3 - affects muscle fiber type distribution'},
                'rs4644994': {'risk_allele': 'A', 'weight': 0.8,
                             'description': 'IGF-1 - influences muscle development'}
            },
            'Exercise Recovery': {
                'rs1049434': {'risk_allele': 'T', 'weight': 1.0,
                             'description': 'MCT1 gene - affects lactate clearance'},
                'rs8192678': {'risk_allele': 'A', 'weight': 0.9,
                             'description': 'PPARGC1A - influences recovery speed'}
            },
            'Injury Risk': {
                'rs12722': {'risk_allele': 'C', 'weight': 1.0,
                           'description': 'COL5A1 - affects tendon flexibility'},
                'rs1800012': {'risk_allele': 'T', 'weight': 0.9,
                             'description': 'COL1A1 - influences ligament strength'}
            }
        }
    }
    
    def __init__(self, genome_file: str):
        self.validator = GenomeDataValidator()
        self.api_client = APIClient()
        self.genome_file = genome_file
        self._cached_data = None
        self._cached_health_traits = None
        self._cached_ancestry = None
        
        # Validate and load genome data
        valid, message = self.validator.validate_file(genome_file)
        if not valid:
            raise ValueError(f"Invalid genome file: {message}")
    
    @property
    def data(self) -> pd.DataFrame:
        """Lazy loading of genome data"""
        if self._cached_data is None:
            self._cached_data = self._load_genome_data()
        return self._cached_data
    
    @property
    def health_traits(self) -> Dict:
        """Lazy loading of health traits analysis"""
        if self._cached_health_traits is None:
            self._cached_health_traits = self._analyze_health_traits()
        return self._cached_health_traits
    
    @property
    def ancestry(self) -> Dict:
        """Lazy loading of ancestry analysis"""
        if self._cached_ancestry is None:
            self._cached_ancestry = self._analyze_ancestry()
        return self._cached_ancestry
    
    def _load_genome_data(self) -> pd.DataFrame:
        """Load and process genome data"""
        chunks = []
        total_variants = 0
        
        try:
            for chunk in pd.read_csv(self.genome_file, 
                                   sep='\t',
                                   names=['rsid', 'chromosome', 'position', 'genotype'],
                                   dtype={'rsid': str, 'chromosome': str, 'position': 'Int64', 'genotype': str},
                                   comment='#',
                                   chunksize=CHUNK_SIZE):
                
                # Clean data
                chunk = chunk.assign(
                    rsid=chunk['rsid'].str.strip(),
                    chromosome=chunk['chromosome'].str.strip().str.upper(),
                    genotype=chunk['genotype'].str.strip().str.upper()
                )
                
                # Filter valid rows with updated genotype pattern
                chunk = chunk[
                    chunk['rsid'].str.match(RSID_PATTERN, na=False) &
                    chunk['chromosome'].isin(VALID_CHROMOSOMES) &
                    (
                        chunk['genotype'].isin(['--', 'D', 'I', 'DD', 'II', 'DI']) |  # Special genotypes
                        chunk['genotype'].str.match(r'^[ACGT]{1,2}$', na=False)  # Standard nucleotides
                    )
                ]
                
                total_variants += len(chunk)
                chunks.append(chunk)
                logging.info(f"Processed {total_variants:,} variants...")
        
        except Exception as e:
            raise ValueError(f"Error loading genome data: {str(e)}")
        
        if not chunks:
            raise ValueError("No valid variants found in file")
        
        df = pd.concat(chunks, ignore_index=True)
        logging.info(f"Successfully loaded {len(df):,} variants")
        return df
    
    def _analyze_health_traits(self) -> Dict:
        """Analyze health traits based on genetic markers"""
        results = {}
        for category, traits in self.TRAIT_DEFINITIONS.items():
            results[category] = {}
            for trait, snps in traits.items():
                matches = 0
                total_weight = 0
                details = []
                
                for rsid, info in snps.items():
                    try:
                        variant = self.data[self.data['rsid'] == rsid]
                        if not variant.empty:
                            genotype = variant['genotype'].iloc[0]
                            if genotype != '--':
                                weight = float(info['weight'])
                                has_risk = info['risk_allele'] in genotype
                                risk_count = genotype.count(info['risk_allele'])
                                
                                details.append({
                                    'rsid': rsid,
                                    'genotype': genotype,
                                    'risk_allele': info['risk_allele'],
                                    'has_risk': has_risk,
                                    'risk_count': risk_count,
                                    'weight': weight,
                                    'description': info.get('description', 'No description available')
                                })
                                
                                if has_risk:
                                    matches += 1
                                    total_weight += weight * risk_count
                    except Exception as e:
                        logging.warning(f"Error analyzing trait {trait} for SNP {rsid}: {str(e)}")
                        continue
                
                if details:
                    score = total_weight / (len(snps) * 2)  # Normalize to 0-1
                    results[category][trait] = {
                        'score': score,
                        'status': self._interpret_risk_score(score),
                        'confidence': 'High' if len(details) == len(snps) else 'Moderate',
                        'details': details
                    }
                else:
                    results[category][trait] = {
                        'score': 0,
                        'status': 'Unknown',
                        'confidence': 'Low',
                        'details': []
                    }
        
        return results
    
    def _interpret_risk_score(self, score: float) -> str:
        """Convert numerical score to risk category"""
        if score > 0.7:
            return 'High Risk'
        elif score > 0.3:
            return 'Moderate Risk'
        else:
            return 'Low Risk'
    
    def _analyze_ancestry(self) -> Dict:
        """Analyze ancestry composition"""
        ancestry_markers = {
            'European': {
                'rs4988235': {'allele': 'T', 'weight': 1.0},
                'rs1426654': {'allele': 'A', 'weight': 1.0}
            },
            'East Asian': {
                'rs3811801': {'allele': 'A', 'weight': 1.0},
                'rs671': {'allele': 'A', 'weight': 1.0}
            },
            'African': {
                'rs2814778': {'allele': 'C', 'weight': 1.0},
                'rs1426654': {'allele': 'G', 'weight': 1.0}
            }
        }
        
        results = {
            'regional_ancestry': {},
            'mt_haplogroup': self._determine_haplogroup(),
            'neanderthal_percentage': self._calculate_neanderthal()
        }
        
        # Calculate regional ancestry percentages
        total_score = 0
        scores = {}
        
        for population, markers in ancestry_markers.items():
            population_score = 0
            for rsid, info in markers.items():
                variant = self.data[self.data['rsid'] == rsid]
                if not variant.empty:
                    genotype = variant['genotype'].iloc[0]
                    if info['allele'] in genotype:
                        population_score += info['weight']
            scores[population] = population_score
            total_score += population_score
        
        if total_score > 0:
            results['regional_ancestry'] = {
                pop: (score / total_score) * 100 
                for pop, score in scores.items()
            }
        
        return results
    
    def _determine_haplogroup(self) -> Dict:
        """Determine mitochondrial haplogroup"""
        mt_variants = self.data[self.data['chromosome'] == 'MT']
        
        # Simplified haplogroup determination
        return {
            'haplogroup': 'H',  # Placeholder - would need proper haplogroup algorithm
            'confidence': 0.8,
            'variants_analyzed': len(mt_variants)
        }
    
    def _calculate_neanderthal(self) -> Dict:
        """Calculate Neanderthal ancestry percentage"""
        neanderthal_markers = {
            'rs6679627': 'A',
            'rs3802971': 'G'
        }
        
        matches = 0
        total = len(neanderthal_markers)
        
        for rsid, allele in neanderthal_markers.items():
            variant = self.data[self.data['rsid'] == rsid]
            if not variant.empty and allele in variant['genotype'].iloc[0]:
                matches += 1
        
        return {
            'percentage': (matches / total) * 2.5,  # Approximate scaling
            'confidence': 'Moderate',
            'markers_found': matches,
            'total_markers': total
        }
    
    def generate_report(self, output_file: str = "dna_report.md"):
        """Generate comprehensive markdown report"""
        if output_file.endswith('.pdf'):
            output_file = output_file.replace('.pdf', '.md')
        
        with open(output_file, 'w') as f:
            # Header
            f.write(f"# DNA Analysis Report\n")
            f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Cool Findings Summary
            f.write("## 🌟 Cool Findings\n")
            cool_findings = []
            
            # Check for interesting traits with high scores
            for category, traits in self.health_traits.items():
                for trait, info in traits.items():
                    if info['score'] > 0.7:  # High likelihood
                        if category == 'Fun & Unique Traits':
                            cool_findings.append(f"- **{trait}**: {self._get_trait_description(category, trait)}")
                    elif info['score'] < 0.3:  # Low likelihood but might be interesting
                        if trait in ['Cilantro Taste Perception', 'Perfect Pitch']:
                            cool_findings.append(f"- **{trait}**: {self._get_trait_description(category, trait)}")
            
            if cool_findings:
                f.write("Here are some of your most interesting genetic traits:\n\n")
                f.write("\n".join(cool_findings) + "\n\n")
            else:
                f.write("Your genetic profile contains many common variants. "+
                       "Check the detailed sections below for more insights.\n\n")
            
            # Summary Statistics
            f.write("## Summary Statistics\n")
            f.write(f"- Total variants analyzed: {len(self.data):,}\n")
            f.write(f"- Chromosomes covered: {len(self.data['chromosome'].unique())}\n\n")
            
            # Health Traits Analysis
            for category, traits in self.health_traits.items():
                f.write(f"## {category}\n")
                for trait, info in traits.items():
                    f.write(f"### {trait}\n")
                    f.write(f"- Status: {info['status']}\n")
                    f.write(f"- Confidence: {info['confidence']}\n")
                    
                    # Add detailed variant information
                    if info['details']:
                        f.write("\nRelevant Variants:\n")
                        for variant in info['details']:
                            f.write(f"- {variant['rsid']}: {variant['genotype']}\n")
                            # Add description if available
                            if category in self.TRAIT_DEFINITIONS and trait in self.TRAIT_DEFINITIONS[category]:
                                trait_info = self.TRAIT_DEFINITIONS[category][trait]
                                if variant['rsid'] in trait_info:
                                    f.write(f"  - {trait_info[variant['rsid']]['description']}\n")
                    f.write("\n")
            
            # Ancestry Analysis
            f.write("## Ancestry Composition\n")
            for population, percentage in self.ancestry['regional_ancestry'].items():
                f.write(f"- {population}: {percentage:.1f}%\n")
            
            # Haplogroup Information
            haplogroup = self.ancestry['mt_haplogroup']
            f.write(f"\n### Maternal Haplogroup\n")
            f.write(f"- Haplogroup: {haplogroup['haplogroup']}\n")
            f.write(f"- Confidence: {haplogroup['confidence']:.1%}\n")
            
            # Neanderthal Ancestry
            neanderthal = self.ancestry['neanderthal_percentage']
            f.write(f"\n### Neanderthal Ancestry\n")
            f.write(f"- Percentage: {neanderthal['percentage']:.1f}%\n")
            f.write(f"- Confidence: {neanderthal['confidence']}\n")
            
            # Environmental Recommendations
            f.write("\n## 🌍 Environmental Adaptations\n")
            env_traits = self.health_traits.get('Environmental Response', {})
            
            # UV Sensitivity
            if 'UV Sensitivity' in env_traits:
                uv_info = env_traits['UV Sensitivity']
                f.write("\n### Sun Exposure\n")
                if uv_info['score'] > 0.5:
                    f.write("- You may have increased UV sensitivity\n")
                    f.write("- Recommended actions:\n")
                    f.write("  - Use broad-spectrum sunscreen (SPF 30+)\n")
                    f.write("  - Seek shade during peak hours (10am-4pm)\n")
                    f.write("  - Regular skin checks\n")
                else:
                    f.write("- You appear to have typical UV response\n")
                    f.write("- Still recommended to use sun protection\n")
            
            # Temperature Adaptation
            if 'Temperature Adaptation' in env_traits:
                temp_info = env_traits['Temperature Adaptation']
                f.write("\n### Temperature Response\n")
                if temp_info['score'] > 0.5:
                    f.write("- You may be more sensitive to temperature changes\n")
                    f.write("- Recommendations:\n")
                    f.write("  - Gradual adaptation to temperature changes\n")
                    f.write("  - Extra precautions in extreme weather\n")
                else:
                    f.write("- You likely adapt well to temperature changes\n")
            
            # Lifestyle Recommendations
            f.write("\n## 💪 Personalized Lifestyle Recommendations\n")
            
            # Exercise timing based on circadian rhythm
            if 'Circadian Rhythm' in self.health_traits.get('Fun & Unique Traits', {}):
                rhythm_info = self.health_traits['Fun & Unique Traits']['Circadian Rhythm']
                f.write("\n### Optimal Exercise Timing\n")
                if rhythm_info['score'] > 0.5:
                    f.write("- Your genetics suggest you're more of a 'morning person'\n")
                    f.write("- Consider scheduling workouts in the morning\n")
                else:
                    f.write("- Your genetics suggest you're more of a 'night owl'\n")
                    f.write("- You might perform better with evening workouts\n")
            
            # Nutrition timing
            f.write("\n### Nutrition Timing\n")
            if 'Carbohydrate Response' in self.health_traits.get('Nutrition & Diet', {}):
                carb_info = self.health_traits['Nutrition & Diet']['Carbohydrate Response']
                if carb_info['score'] > 0.5:
                    f.write("- Consider timing carbohydrate intake around exercise\n")
                    f.write("- May benefit from post-workout nutrition within 30 minutes\n")
                else:
                    f.write("- More flexible window for post-exercise nutrition\n")
            
            # Recovery recommendations
            if 'Exercise Recovery' in self.health_traits.get('Exercise Response', {}):
                recovery_info = self.health_traits['Exercise Response']['Exercise Recovery']
                f.write("\n### Recovery Strategy\n")
                if recovery_info['score'] > 0.5:
                    f.write("- You may need longer recovery periods between intense workouts\n")
                    f.write("- Focus on:\n")
                    f.write("  - Quality sleep (7-9 hours)\n")
                    f.write("  - Active recovery days\n")
                    f.write("  - Proper hydration\n")
                else:
                    f.write("- You likely recover relatively quickly from exercise\n")
                    f.write("- Can handle more frequent training sessions\n")
            
            # Recommendations
            f.write("\n## Personalized Recommendations\n")
            
            # Fitness recommendations
            f.write("\n### Fitness\n")
            muscle_trait = self.health_traits['Exercise Response']['Muscle Type']
            endurance_trait = self.health_traits['Exercise Response']['Exercise Recovery']
            
            if muscle_trait['score'] > endurance_trait['score']:
                f.write("- Your genetics may favor strength-based activities\n")
                f.write("- Consider incorporating weight training into your routine\n")
            else:
                f.write("- Your genetics may favor endurance activities\n")
                f.write("- Consider activities like running, cycling, or swimming\n")
            
            # Diet recommendations
            f.write("\n### Diet\n")
            caffeine_info = self.health_traits['Nutrition & Diet']['Caffeine Metabolism']
            if caffeine_info['score'] > 0.5:
                f.write("- You may be a slow caffeine metabolizer\n")
                f.write("- Consider limiting caffeine intake, especially in the evening\n")
            else:
                f.write("- You appear to metabolize caffeine efficiently\n")
                f.write("- Regular coffee consumption may be well-tolerated\n")
            
            lactose_info = self.health_traits['Nutrition & Diet']['Carbohydrate Response']
            if lactose_info['score'] > 0.5:
                f.write("- You likely have good lactose tolerance\n")
            else:
                f.write("- You may have reduced lactose tolerance\n")
                f.write("- Consider lactose-free alternatives or digestive enzymes\n")
        
        logging.info(f"Generated report: {output_file}")
    
    def run_dashboard(self):
        """Launch interactive dashboard"""
        app = DashApp(self)
        app.run()
    
    def _get_trait_description(self, category: str, trait: str) -> str:
        """Get a user-friendly description for a trait"""
        try:
            trait_info = self.TRAIT_DEFINITIONS[category][trait]
            # Get the description from the first SNP (they usually share the same general description)
            first_snp = next(iter(trait_info.values()))
            return first_snp['description'].split(' - ')[0]  # Take just the first part of the description
        except:
            return "Interesting genetic variant"

class DashApp:
    """Interactive dashboard for DNA analysis"""
    
    def __init__(self, analyzer: UnifiedDNAAnalyzer):
        self.app = dash.Dash(__name__)
        self.analyzer = analyzer
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.H1('DNA Analysis Dashboard', style={'textAlign': 'center'}),
            
            # Tabs for different analyses
            dcc.Tabs([
                # Variant Browser
                dcc.Tab(label='Variant Browser', children=[
                    html.Div([
                        dcc.Dropdown(
                            id='chromosome-selector',
                            options=[{'label': f'Chromosome {c}', 'value': c} 
                                   for c in sorted(self.analyzer.data['chromosome'].unique())],
                            value='1'
                        ),
                        dcc.Graph(id='variant-plot'),
                        dash_table.DataTable(
                            id='variant-table',
                            columns=[
                                {'name': 'RSID', 'id': 'rsid'},
                                {'name': 'Position', 'id': 'position'},
                                {'name': 'Genotype', 'id': 'genotype'}
                            ],
                            page_size=10
                        )
                    ])
                ]),
                
                # Health Insights
                dcc.Tab(label='Health Insights', children=[
                    html.Div([
                        dcc.Graph(id='health-plot'),
                        html.Div(id='health-details')
                    ])
                ]),
                
                # Ancestry
                dcc.Tab(label='Ancestry', children=[
                    html.Div([
                        dcc.Graph(id='ancestry-plot'),
                        html.Div(id='ancestry-details')
                    ])
                ])
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('variant-plot', 'figure'),
             Output('variant-table', 'data')],
            [Input('chromosome-selector', 'value')]
        )
        def update_variant_view(chromosome):
            # Filter data for selected chromosome
            chr_data = self.analyzer.data[self.analyzer.data['chromosome'] == chromosome]
            
            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chr_data['position'],
                y=[1] * len(chr_data),
                mode='markers',
                marker=dict(
                    size=8,
                    color=chr_data['genotype'].map(lambda x: len(set(x))),
                    colorscale='Viridis'
                ),
                text=chr_data.apply(
                    lambda row: f"RSID: {row['rsid']}<br>Position: {row['position']}<br>Genotype: {row['genotype']}", 
                    axis=1
                ),
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title=f'Chromosome {chromosome} Variants',
                xaxis_title='Position',
                yaxis_visible=False,
                height=400
            )
            
            # Update table data
            table_data = chr_data.to_dict('records')
            
            return fig, table_data
        
        @self.app.callback(
            Output('health-plot', 'figure'),
            [Input('health-details', 'children')]  # Dummy input for initial load
        )
        def update_health_plot(_):
            data = []
            for category, traits in self.analyzer.health_traits.items():
                for trait, info in traits.items():
                    data.append({
                        'Category': category,
                        'Trait': trait,
                        'Score': info['score'] * 100,
                        'Status': info['status']
                    })
            
            df = pd.DataFrame(data)
            fig = px.bar(
                df,
                x='Trait',
                y='Score',
                color='Category',
                title='Health Traits Analysis',
                labels={'Score': 'Risk Score (%)'}
            )
            
            return fig
        
        @self.app.callback(
            Output('ancestry-plot', 'figure'),
            [Input('ancestry-details', 'children')]  # Dummy input for initial load
        )
        def update_ancestry_plot(_):
            data = self.analyzer.ancestry['regional_ancestry']
            fig = go.Figure(data=[go.Pie(
                labels=list(data.keys()),
                values=list(data.values()),
                hole=.3
            )])
            
            fig.update_layout(
                title='Ancestry Composition',
                height=400
            )
            
            return fig
    
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive DNA Analysis Tool")
    parser.add_argument("--genome", type=str, required=True, help="Path to genome data file")
    parser.add_argument("--report", type=str, default="dna_report.md", 
                       help="Output path for report (supports .md or .pdf)")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching the dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port number")
    
    args = parser.parse_args()
    
    try:
        analyzer = UnifiedDNAAnalyzer(args.genome)
        
        # Generate report in requested format
        analyzer.generate_report(args.report)
        print(f"\nReport generated: {args.report}")
        
        if not args.no_dashboard:
            print(f"\nLaunching dashboard on port {args.port}... Press Ctrl+C to exit")
            analyzer.run_dashboard()
            
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()