from flask import Flask, render_template, request, jsonify
import os
from dna_analyzer import DNAAnalyzer
from file_handler import FileHandler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        # Save file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process file
        data, metadata = FileHandler.read_file(file_path)
        analyzer = DNAAnalyzer(data, metadata)
        
        # Get analysis results
        basic_stats = analyzer.get_basic_stats()
        composition = analyzer.analyze_sequence_composition()
        validation = analyzer.validate_sequences()
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'status': 'success',
            'basic_stats': basic_stats,
            'composition': composition,
            'validation': validation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_pattern():
    try:
        pattern = request.json.get('pattern', '')
        if not pattern:
            return jsonify({'error': 'No pattern provided'}), 400
            
        # Since we don't store files permanently, return empty results
        return jsonify({'matches': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)