"""
Kiểm thử tích hợp TruLens vào hệ thống Text-to-SQL
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import module Text2SQLProcessor với TruLens
from models.text2sql_processor_trulens import Text2SQLProcessor

class TestTruLensIntegration(unittest.TestCase):
    """Test case cho việc tích hợp TruLens vào Text2SQLProcessor"""
    
    def setUp(self):
        """Thiết lập trước mỗi test case"""
        # Tạo schema mẫu
        self.schema = """
        {
            "tables": [
                {
                    "name": "sales",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "ID giao dịch"},
                        {"name": "product_id", "type": "INTEGER", "description": "ID sản phẩm"},
                        {"name": "quantity", "type": "INTEGER", "description": "Số lượng bán"},
                        {"name": "price", "type": "FLOAT", "description": "Giá bán"},
                        {"name": "date", "type": "DATE", "description": "Ngày bán"},
                        {"name": "branch_id", "type": "INTEGER", "description": "ID chi nhánh"}
                    ]
                },
                {
                    "name": "products",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "ID sản phẩm"},
                        {"name": "name", "type": "TEXT", "description": "Tên sản phẩm"},
                        {"name": "category", "type": "TEXT", "description": "Danh mục sản phẩm"},
                        {"name": "cost", "type": "FLOAT", "description": "Giá vốn"}
                    ]
                },
                {
                    "name": "branches",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "ID chi nhánh"},
                        {"name": "name", "type": "TEXT", "description": "Tên chi nhánh"},
                        {"name": "city", "type": "TEXT", "description": "Thành phố"}
                    ]
                }
            ]
        }
        """
        
        # Tạo câu hỏi mẫu
        self.query = "Hiển thị tổng doanh thu theo từng chi nhánh"
        
        # Tạo SQL mẫu
        self.sql = """
        SELECT b.name AS branch_name, SUM(s.quantity * s.price) AS total_revenue
        FROM sales s
        JOIN branches b ON s.branch_id = b.id
        GROUP BY b.name
        ORDER BY total_revenue DESC
        """
        
        # Tạo instance của Text2SQLProcessor
        self.processor = Text2SQLProcessor()
    
    def test_processor_initialization(self):
        """Kiểm tra khởi tạo Text2SQLProcessor"""
        self.assertIsNotNone(self.processor)
        print(f"TruLens available: {self.processor.trulens_available}")
        print(f"Gemini available: {self.processor.gemini_available}")
    
    def test_clean_sql(self):
        """Kiểm tra hàm clean_sql"""
        # SQL với markdown formatting
        sql_with_markdown = "```sql\nSELECT * FROM table\n```"
        cleaned_sql = self.processor.clean_sql(sql_with_markdown)
        self.assertEqual(cleaned_sql, "SELECT * FROM table")
        
        # SQL với khoảng trắng thừa
        sql_with_whitespace = "  SELECT * FROM table  "
        cleaned_sql = self.processor.clean_sql(sql_with_whitespace)
        self.assertEqual(cleaned_sql, "SELECT * FROM table")
    
    @patch('models.text2sql_processor_trulens.genai')
    def test_generate_sql(self, mock_genai):
        """Kiểm tra hàm generate_sql với mock Gemini"""
        # Mock response từ Gemini
        mock_response = MagicMock()
        mock_response.text = "SQL: SELECT * FROM table\nExplanation: This is a test explanation"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Set gemini_available = True để test
        self.processor.gemini_available = True
        
        # Gọi hàm generate_sql
        result = self.processor.generate_sql(self.query, self.schema)
        
        # Kiểm tra kết quả
        self.assertTrue(result["success"])
        self.assertEqual(result["sql"], "SELECT * FROM table")
        self.assertEqual(result["explanation"], "This is a test explanation")
    
    def test_evaluate_sql_without_trulens(self):
        """Kiểm tra hàm evaluate_sql khi TruLens không khả dụng"""
        # Đảm bảo trulens_available = False
        self.processor.trulens_available = False
        
        # Gọi hàm evaluate_sql
        result = self.processor.evaluate_sql(self.query, self.sql, self.schema)
        
        # Kiểm tra kết quả
        self.assertTrue(result["success"])
        self.assertFalse(result["metrics_available"])
        self.assertEqual(result["evaluation_framework"], "TruLens (simulated)")
    
    @unittest.skipIf(not os.environ.get("OPENAI_API_KEY"), "OpenAI API key not set")
    def test_evaluate_sql_with_trulens(self):
        """Kiểm tra hàm evaluate_sql với TruLens thật (chỉ chạy nếu có API key)"""
        # Kiểm tra nếu TruLens khả dụng
        if not self.processor.trulens_available:
            self.skipTest("TruLens not available")
        
        # Gọi hàm evaluate_sql
        result = self.processor.evaluate_sql(self.query, self.sql, self.schema)
        
        # Kiểm tra kết quả
        self.assertTrue(result["success"])
        self.assertTrue(result["metrics_available"])
        self.assertEqual(result["evaluation_framework"], "TruLens")
        
        # Kiểm tra các chỉ số
        self.assertIn("relevancy_score", result)
        self.assertIn("hallucination_score", result)
        self.assertIn("contextual_relevancy_score", result)
        self.assertIn("correctness_score", result)
        self.assertIn("syntax_score", result)
        
        # In ra các chỉ số để kiểm tra
        print("\nTruLens evaluation results:")
        print(f"Relevancy: {result.get('relevancy_score', 'N/A')}")
        print(f"Hallucination: {result.get('hallucination_score', 'N/A')}")
        print(f"Contextual Relevancy: {result.get('contextual_relevancy_score', 'N/A')}")
        print(f"Correctness: {result.get('correctness_score', 'N/A')}")
        print(f"Syntax: {result.get('syntax_score', 'N/A')}")
    
    @patch('src.models.text2sql_processor_trulens.genai')
    def test_process_query(self, mock_genai):
        """Kiểm tra quy trình xử lý câu hỏi hoàn chỉnh"""
        # Mock response từ Gemini
        mock_response = MagicMock()
        mock_response.text = "SQL: SELECT * FROM table\nExplanation: This is a test explanation"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Set gemini_available = True để test
        self.processor.gemini_available = True
        
        # Mock hàm evaluate_sql
        original_evaluate_sql = self.processor.evaluate_sql
        self.processor.evaluate_sql = MagicMock(return_value={
            "success": True,
            "all_passed": True,
            "metrics_available": True,
            "relevancy_score": 0.85,
            "hallucination_score": 0.15,
            "contextual_relevancy_score": 0.80,
            "correctness_score": 0.90,
            "syntax_score": 0.95,
            "evaluation_framework": "TruLens (mock)"
        })
        
        # Gọi hàm process_query
        result = self.processor.process_query(self.query, self.schema)
        
        # Khôi phục hàm evaluate_sql
        self.processor.evaluate_sql = original_evaluate_sql
        
        # Kiểm tra kết quả
        self.assertEqual(result["original_query"], self.query)
        self.assertEqual(result["processed_query"], self.query)
        self.assertEqual(result["source"], "generated")
        self.assertEqual(result["sql"], "SELECT * FROM table")
        self.assertEqual(result["explanation"], "This is a test explanation")
        
        # Kiểm tra evaluation
        evaluation = result["evaluation"]
        self.assertTrue(evaluation["success"])
        self.assertTrue(evaluation["all_passed"])
        self.assertTrue(evaluation["metrics_available"])
        self.assertEqual(evaluation["relevancy_score"], 0.85)
        self.assertEqual(evaluation["hallucination_score"], 0.15)
        self.assertEqual(evaluation["contextual_relevancy_score"], 0.80)
        self.assertEqual(evaluation["correctness_score"], 0.90)
        self.assertEqual(evaluation["syntax_score"], 0.95)
        self.assertEqual(evaluation["evaluation_framework"], "TruLens (mock)")
    
    def test_detect_ambiguity(self):
        """Kiểm tra phát hiện tính mơ hồ trong câu hỏi"""
        # Câu hỏi mơ hồ
        ambiguous_query = "Hiển thị các sản phẩm tốt nhất"
        is_ambiguous, analysis = self.processor.detect_ambiguity(ambiguous_query, self.schema)
        self.assertTrue(is_ambiguous)
        
        # Câu hỏi rõ ràng
        clear_query = "Hiển thị tất cả sản phẩm có giá trên 100"
        is_ambiguous, analysis = self.processor.detect_ambiguity(clear_query, self.schema)
        self.assertFalse(is_ambiguous)

if __name__ == '__main__':
    unittest.main()
