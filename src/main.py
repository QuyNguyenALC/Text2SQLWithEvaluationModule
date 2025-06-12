"""
Main entry point for the Flask application with updated DeepEval metrics display
"""

import os
import sys
import tempfile
from flask import Flask, render_template, send_from_directory, session
from routes.api_trulens import api_bp

# Create Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Routes
@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('static', 'index_trulens.html')

@app.route('/<path:path>')
def static_files(path):
    """Serve static files"""
    return send_from_directory('static', path)

# Main entry point
if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Print API status
    from models.text2sql_processor_trulens import Text2SQLProcessor
    processor = Text2SQLProcessor()
    
    print("\n=== Text-to-SQL Demo API Status ===")
    print(f"Gemini API: {'Available' if processor.gemini_available else 'Not Available'}")
    print(f"trulens: {'Available' if processor.trulens_available else 'Not Available'}")
    if not processor.trulens_available and processor.openai_api_key_available:
        print("  - trulens installed but OpenAI API key not set")
    elif not processor.trulens_available:
        print("  - trulens not installed or OpenAI API key not set")
    print("===================================\n")
    
    # Run the app
    app.run(host='0.0.0.0', port=5001, debug=True)