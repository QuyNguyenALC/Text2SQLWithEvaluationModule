"""
API routes for the Text-to-SQL Demo with TruLens integration
"""

from flask import Blueprint, request, jsonify, current_app
import os
import json
import pandas as pd
import sqlite3
import traceback
from models.text2sql_processor_trulens import Text2SQLProcessor
import numpy as np

# Khởi tạo blueprint
api_bp = Blueprint('api', __name__)

# Khởi tạo Text2SQLProcessor
processor = Text2SQLProcessor()

# Đường dẫn đến database
DB_PATH = 'src/database/grocery.db'
print(f"Checking database at: {DB_PATH}")
print(f"Database file exists: {os.path.exists(DB_PATH)}")

@api_bp.route('/status', methods=['GET'])
def get_status():
    """Trả về trạng thái của các API và dịch vụ"""
    return jsonify({
        'gemini_available': processor.gemini_available,
        'trulens_available': processor.trulens_available,
        'trulens_message': 'Using TruLens for SQL evaluation' if processor.trulens_available else 'OpenAI API key not set or TruLens not installed'
    })

@api_bp.route('/load/data', methods=['GET'])
def load_data():
    """Đọc dữ liệu từ database"""
    try:
        print(f"Attempting to connect to database at: {DB_PATH}")
        
        if not os.path.exists(DB_PATH):
            return jsonify({
                'success': False,
                'error': f'Database file not found at: {DB_PATH}'
            })
        
        # Kết nối đến database
        conn = sqlite3.connect(DB_PATH)
        
        # Kiểm tra bảng có tồn tại không
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='grocery_sales'")
        if not cursor.fetchone():
            return jsonify({
                'success': False,
                'error': 'Table grocery_sales does not exist in database'
            })
        
        # Đọc dữ liệu từ bảng grocery_sales
        df = pd.read_sql_query("SELECT * FROM grocery_sales", conn)
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No data found in grocery_sales table'
            })
        
        print(f"Successfully loaded {len(df)} rows from database")
        
        # Lưu dữ liệu vào session
        current_app.config['DATA_DF'] = df
        
        # Chuyển NaN thành None để JSON hóa hợp lệ
        preview = df.head(5).replace({np.nan: None}).to_dict(orient='records')
        
        response_data = {
            'success': True,
            'preview': preview,
            'rows': len(df),
            'columns': list(df.columns)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc()
        })
    finally:
        if 'conn' in locals():
            conn.close()

@api_bp.route('/load/schema', methods=['GET'])
def load_schema():
    """Đọc schema từ file JSON"""
    try:
        # Đọc schema từ file
        schema_path = 'grocery_sales_schema.json'
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Lưu schema vào session
        current_app.config['SCHEMA'] = schema
        
        return jsonify({
            'success': True,
            'preview': schema
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@api_bp.route('/query', methods=['POST'])
def process_query():
    """Xử lý câu hỏi ngôn ngữ tự nhiên"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    # Kiểm tra dữ liệu và schema đã được load chưa
    if 'SCHEMA' not in current_app.config:
        return jsonify({'success': False, 'error': 'Schema not loaded'})
    
    schema = current_app.config['SCHEMA']
    schema_str = json.dumps(schema, ensure_ascii=False)
    
    try:
        # Xử lý câu hỏi
        result = processor.process_query(query, schema_str)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@api_bp.route('/execute', methods=['POST'])
def execute_sql():
    """Thực thi câu lệnh SQL trên dữ liệu từ database"""
    data = request.json
    sql = data.get('sql', '')
    
    if not sql:
        return jsonify({'success': False, 'error': 'No SQL provided'})
    
    try:
        # Kết nối đến database
        conn = sqlite3.connect(DB_PATH)
        
        # Làm sạch SQL
        sql = processor.clean_sql(sql)
        
        # Thực thi SQL trên database
        result = pd.read_sql_query(sql, conn)
        
        return jsonify({
            'success': True,
            'results': result.to_dict(orient='records'),
            'rows': len(result),
            'columns': list(result.columns)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
    finally:
        conn.close()

@api_bp.route('/clarify', methods=['POST'])
def process_clarification():
    """
    Xử lý câu hỏi sau khi làm rõ (clarification).
    """
    data = request.json
    original_query = data.get('original_query', '')
    clarification_responses = data.get('clarification_responses', {})

    if not original_query or not clarification_responses:
        return jsonify({'success': False, 'error': 'Missing required parameters'})

    # Kiểm tra dữ liệu và schema đã được load chưa
    if 'SCHEMA' not in current_app.config:
        return jsonify({'success': False, 'error': 'Schema not loaded'})

    schema = current_app.config['SCHEMA']
    schema_str = json.dumps(schema, ensure_ascii=False)

    try:
        # Xử lý câu hỏi đã làm rõ
        result = processor.process_clarified_query(original_query, clarification_responses, schema_str)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
