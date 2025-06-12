"""
Data Manager Module - Handles data loading, schema management, and data operations
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union

class DataManager:
    """Manages data loading, schema management, and data operations for the Text-to-SQL Demo."""
    
    def __init__(self):
        """Initialize the Data Manager."""
        self.data = None
        self.schema = None
        self.data_path = None
        self.schema_path = None
    
    def load_data(self, file_path: str) -> None:
        """
        Load data from CSV or Parquet file
        
        Args:
            file_path: Path to the data file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Store the file path
        self.data_path = file_path
        
        # Determine file type from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_ext in ['.parquet', '.pq']:
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def load_schema(self, file_path: str) -> None:
        """
        Load schema from JSON file
        
        Args:
            file_path: Path to the schema file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Store the file path
        self.schema_path = file_path
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading schema: {str(e)}")
    
    def get_data_preview(self, num_rows: int = 5) -> List[Dict]:
        """
        Get a preview of the data
        
        Args:
            num_rows: Number of rows to preview
            
        Returns:
            List of dictionaries representing the data preview
        """
        if self.data is None:
            return []
        
        # Get the first num_rows rows
        preview = self.data.head(num_rows)
        
        # Convert to list of dictionaries
        return preview.to_dict('records')
    
    def get_schema_preview(self) -> str:
        """
        Get a preview of the schema
        
        Returns:
            String representation of the schema
        """
        if self.schema is None:
            return "No schema available"
        
        return json.dumps(self.schema, indent=2)
    
    def infer_schema_from_data(self) -> Dict:
        """
        Infer schema from data
        
        Returns:
            Dictionary representing the inferred schema
        """
        if self.data is None:
            return {}
        
        # Get column names and types
        columns = []
        for col_name, dtype in self.data.dtypes.items():
            col_type = "TEXT"
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "REAL"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                col_type = "DATETIME"
            
            columns.append({
                "name": col_name,
                "type": col_type,
                "description": f"Column {col_name}"
            })
        
        # Create schema
        schema = {
            "tables": [
                {
                    "name": "data",
                    "description": "Uploaded data",
                    "columns": columns
                }
            ]
        }
        
        return schema
    
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query on the data
        
        Args:
            sql: SQL query to execute
            
        Returns:
            DataFrame containing the query results
        """
        if self.data is None:
            raise Exception("No data loaded")
        
        try:
            # Use pandas to execute SQL on the data
            import sqlite3
            from sqlalchemy import create_engine
            
            # Create in-memory SQLite database
            conn = sqlite3.connect(':memory:')
            
            # Write data to database
            self.data.to_sql('data', conn, index=False)
            
            # Execute query
            results = pd.read_sql_query(sql, conn)
            
            return results
        except Exception as e:
            raise Exception(f"Error executing SQL: {str(e)}")