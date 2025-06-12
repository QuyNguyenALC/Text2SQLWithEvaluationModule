# """
# Text-to-SQL Processor Module - TruLens Integration
# Thay thế DeepEval bằng TruLens để đánh giá chất lượng SQL
# """

# import os
# import json
# import time
# import numpy as np
# import re
# from typing import Dict, List, Tuple, Optional, Any, Union
# import google.generativeai as genai

# # Cài đặt TruLens
# try:
#     # from trulens_eval import Tru, Feedback
#     from trulens.core import TruSession
#     from trulens.core import Feedback, Select
#     # from trulens.core.Feedback import Groundedness, Relevance, ContextRelevance
#     # from trulens.feedback.feedback import Groundedness, Relevance, ContextRelevance
#     # from trulens_eval.feedback import Groundedness, Relevance, ContextRelevance
#     from trulens.providers.openai import OpenAI as TruOpenAI
#     openai_provider = TruOpenAI()
#     # from trulens_eval.feedback.provider import OpenAI
#     TRULENS_AVAILABLE = True
# except ImportError:
#     TRULENS_AVAILABLE = False

"""
Text-to-SQL Processor Module - TruLens Integration
Thay thế DeepEval bằng TruLens để đánh giá chất lượng SQL
"""

import os
import json
import time
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import google.generativeai as genai

# Custom Feedback Functions
class Relevance:
    """Đánh giá độ liên quan giữa câu hỏi và câu trả lời"""
    def __init__(self, openai_provider):
        self.openai = openai_provider
        
    def score(self, query: str, response: str) -> float:
        prompt = f"""
        Evaluate how relevant the response is to the query on a scale from 0 to 1.
        Query: {query}
        Response: {response}
        Score (0-1):
        """
        try:
            result = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluator that returns only a number between 0 and 1."},
                    {"role": "user", "content": prompt}
                ]
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            print(f"Error in Relevance scoring: {str(e)}")
            return 0.0

class Groundedness:
    """Đánh giá độ chính xác của câu trả lời dựa trên context"""
    def __init__(self, openai_provider):
        self.openai = openai_provider
        
    def score(self, response: str, context: List[str]) -> float:
        context_str = "\n".join(context)
        prompt = f"""
        Evaluate how well the response is grounded in the provided context on a scale from 0 to 1.
        Context: {context_str}
        Response: {response}
        Score (0-1):
        """
        try:
            result = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluator that returns only a number between 0 and 1."},
                    {"role": "user", "content": prompt}
                ]
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            print(f"Error in Groundedness scoring: {str(e)}")
            return 0.0

class ContextRelevance:
    """Đánh giá độ liên quan của context với câu hỏi"""
    def __init__(self, openai_provider):
        self.openai = openai_provider
        
    def score(self, query: str, response: str, context: List[str]) -> float:
        context_str = "\n".join(context)
        prompt = f"""
        Evaluate how relevant the context is to both the query and response on a scale from 0 to 1.
        Query: {query}
        Context: {context_str}
        Response: {response}
        Score (0-1):
        """
        try:
            result = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluator that returns only a number between 0 and 1."},
                    {"role": "user", "content": prompt}
                ]
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            print(f"Error in ContextRelevance scoring: {str(e)}")
            return 0.0

# Thêm 2 class feedback functions mới
class SQLCorrectness:
    """Đánh giá độ chính xác của SQL query"""
    def __init__(self, openai_provider):
        self.openai = openai_provider
        
    def score(self, question: str, schema: str, sql: str) -> float:
        prompt = f"""
        Evaluate if the SQL query correctly translates the user's question and follows SQL best practices for the given schema.
        Question: {question}
        Schema: {schema}
        SQL: {sql}
        Score (0-1):
        """
        try:
            result = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluator that returns only a number between 0 and 1."},
                    {"role": "user", "content": prompt}
                ]
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            print(f"Error in SQLCorrectness scoring: {str(e)}")
            return 0.0

class SQLSyntax:
    """Đánh giá cú pháp SQL"""
    def __init__(self, openai_provider):
        self.openai = openai_provider
        
    def score(self, sql: str) -> float:
        prompt = f"""
        Check if the SQL query has correct syntax.
        SQL: {sql}
        Score (0-1):
        """
        try:
            result = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluator that returns only a number between 0 and 1."},
                    {"role": "user", "content": prompt}
                ]
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            print(f"Error in SQLSyntax scoring: {str(e)}")
            return 0.0

# Cài đặt TruLens
try:
    from trulens.core import TruSession
    from trulens.core import Feedback, Select
    from openai import OpenAI  # Thay đổi import này
    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False

class Text2SQLProcessor:
    """Processes natural language queries to SQL using Gemini, FAQ matching, and TruLens."""
    
    def __init__(self, faq_path: Optional[str] = None):
        """
        Initialize the Text2SQL processor
        
        Args:
            faq_path: Path to the FAQ JSON file (optional)
        """
        self.faq_data = self._load_faq(faq_path) if faq_path else []
        self.ambiguous_terms = ["best", "popular", "recent", "top", "important", "tốt nhất", "phổ biến", "gần đây", "hàng đầu", "quan trọng"]
        
        # Configure Gemini if API key is available
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.gemini_available = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_available = True
            except Exception as e:
                print(f"Error configuring Gemini: {str(e)}")
        
        # Configure TruLens if available
        self.openai_api_key_available = os.environ.get("OPENAI_API_KEY") is not None
        self.trulens_available = TRULENS_AVAILABLE and self.openai_api_key_available
        
        if self.trulens_available:
            try:
                # Khởi tạo TruLens
                self.tru = TruSession()
                
                # Khởi tạo OpenAI client
                self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # Thay đổi cách khởi tạo này
                
                # Khởi tạo các feedback functions
                self.relevance = Relevance(self.openai)
                self.groundedness = Groundedness(self.openai)
                self.context_relevance = ContextRelevance(self.openai)
                
                # Custom feedback function cho SQL correctness
                self.sql_correctness = SQLCorrectness(self.openai)
                
                # Custom feedback function cho SQL syntax
                self.sql_syntax = SQLSyntax(self.openai)
                
                # Thiết lập ngưỡng đánh giá
                self.thresholds = {
                    "relevance": 0.7,
                    "groundedness": 0.7,
                    "context_relevance": 0.7,
                    "correctness": 0.6,
                    "syntax": 0.8
                }
                
                print("TruLens initialized successfully")
            except Exception as e:
                print(f"Error initializing TruLens: {str(e)}")
                self.trulens_available = False
        
        # Debug info
        print("TruLens Debug Info:")
        print(f"- TRULENS_AVAILABLE: {TRULENS_AVAILABLE}")
        print(f"- OpenAI API Key available: {self.openai_api_key_available}")
        print(f"- TruLens metrics available: {self.trulens_available}")
        if not TRULENS_AVAILABLE:
            print("  - Reason: TruLens library not installed or import failed")
        elif not self.openai_api_key_available:
            print("  - Reason: OpenAI API key not set in environment variables")
    
    def _load_faq(self, faq_path: str) -> List[Dict]:
        """
        Load FAQ data from JSON file
        
        Args:
            faq_path: Path to the FAQ JSON file
            
        Returns:
            List of FAQ entries
        """
        if not faq_path or not os.path.exists(faq_path):
            return []
        
        try:
            with open(faq_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("questions", [])
        except Exception as e:
            print(f"Error loading FAQ file: {str(e)}")
            return []
    
    def match_faq(self, query: str) -> Optional[Dict]:
        """
        Match query against FAQ entries
        
        Args:
            query: Natural language query
            
        Returns:
            Matched FAQ entry or None
        """
        if not self.faq_data:
            return None
        
        # Simple exact matching for now
        for item in self.faq_data:
            for pattern in item.get("patterns", []):
                if pattern.lower() == query.lower():
                    return item
        
        return None
    
    def detect_ambiguity(self, query: str, schema: str) -> Tuple[bool, Optional[str]]:
        """
        Detect ambiguity in the query
        
        Args:
            query: Natural language query
            schema: Database schema as string
            
        Returns:
            Tuple of (is_ambiguous, ambiguity_analysis)
        """
        # Simple term-based detection
        for term in self.ambiguous_terms:
            if term in query.lower():
                return True, f"Ambiguous term detected: '{term}'"
        
        # Use Gemini for more sophisticated detection if available
        if self.gemini_available:
            try:
                prompt = f"""
                Analyze the following query and determine if it is ambiguous in the context of the given database schema.
                If it is ambiguous, explain why. If it is clear, respond with "The query is clear."
                
                Database Schema:
                {schema}
                
                Query: "{query}"
                
                Analysis:
                """
                
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                
                if "ambiguous" in response.text.lower() or "unclear" in response.text.lower():
                    return True, response.text
                
                return False, None
            except Exception as e:
                print(f"Error using Gemini for ambiguity detection: {str(e)}")
        
        # Default to non-ambiguous if Gemini is not available
        return False, None
    
    def generate_clarification_questions(self, query: str, ambiguity_analysis: str, schema: str) -> str:
        """
        Generate clarification questions for ambiguous queries
        
        Args:
            query: Natural language query
            ambiguity_analysis: Analysis of the ambiguity
            schema: Database schema as string
            
        Returns:
            Clarification questions as string
        """
        if not self.gemini_available:
            # Default questions if Gemini is not available
            return "Could you please clarify your query? What specific information are you looking for?"
        
        try:
            prompt = f"""
            Based on the following ambiguity analysis of a query, generate 1-2 clear, concise questions to help clarify the user's intent.
            For each question, provide 2-3 specific options to choose from.
            
            Database Schema:
            {schema}
            
            Query: "{query}"
            
            Ambiguity Analysis: {ambiguity_analysis}
            
            Format your response as:
            1. [Question]
               - [Option 1]
               - [Option 2]
               - [Option 3]
            
            Clarification Questions:
            """
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Error generating clarification questions: {str(e)}")
            return "Could you please clarify your query? What specific information are you looking for?"
    
    def update_query_with_clarification(self, original_query: str, clarification_responses: Dict[str, str]) -> str:
        """
        Update query based on clarification responses
        
        Args:
            original_query: Original natural language query
            clarification_responses: Dict of clarification responses
            
        Returns:
            Updated query
        """
        if not self.gemini_available:
            # Simple concatenation if Gemini is not available
            clarifications = ", ".join([f"{k}: {v}" for k, v in clarification_responses.items()])
            return f"{original_query} (clarified: {clarifications})"
        
        try:
            prompt = f"""
            Based on the original query and the user's clarification responses, create a clear, unambiguous query.
            
            Original Query: "{original_query}"
            
            Clarification Responses:
            {json.dumps(clarification_responses, indent=2)}
            
            Updated Query:
            """
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Error updating query with clarification: {str(e)}")
            clarifications = ", ".join([f"{k}: {v}" for k, v in clarification_responses.items()])
            return f"{original_query} (clarified: {clarifications})"
    
    def clean_sql(self, sql_text: str) -> str:
        """
        Clean SQL text by removing markdown formatting and extra whitespace
        
        Args:
            sql_text: SQL text that may contain markdown formatting
            
        Returns:
            Clean SQL text
        """
        # Remove markdown code block markers (```sql, ```, etc.)
        sql_text = re.sub(r'```\w*\s*', '', sql_text)
        sql_text = re.sub(r'```\s*$', '', sql_text)
        
        # Remove any leading/trailing whitespace
        sql_text = sql_text.strip()
        
        # Remove any trailing semicolons (optional, depends on your SQL engine)
        # sql_text = re.sub(r';+\s*$', '', sql_text)
        
        return sql_text
    
    def generate_sql(self, query: str, schema: str) -> Dict:
        """
        Generate SQL from natural language query
        
        Args:
            query: Natural language query
            schema: Database schema as string
            
        Returns:
            Dict containing SQL and explanation
        """
        if not self.gemini_available:
            return {
                "success": False,
                "error": "Gemini API is not available. Please set the GOOGLE_API_KEY environment variable."
            }
        
        try:
            prompt = f"""
            You are a Text-to-SQL conversion expert. Convert the following natural language query into a valid SQL query based on the provided schema.
            Also provide a clear explanation of how the SQL query works.
            
            Database Schema:
            {schema}
            
            Query: "{query}"
            
            Respond in the following format:
            SQL: <your SQL query>
            Explanation: <your explanation>
            """
            
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            
            # Parse the response to extract SQL and explanation
            text = response.text
            sql = ""
            explanation = ""
            
            if "SQL:" in text:
                parts = text.split("SQL:", 1)
                if len(parts) > 1:
                    sql_and_explanation = parts[1].strip()
                    if "Explanation:" in sql_and_explanation:
                        sql_parts = sql_and_explanation.split("Explanation:", 1)
                        sql = sql_parts[0].strip()
                        explanation = sql_parts[1].strip()
                    else:
                        sql = sql_and_explanation
            
            # Clean SQL from markdown formatting
            sql = self.clean_sql(sql)
            
            return {
                "success": True,
                "sql": sql,
                "explanation": explanation
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating SQL: {str(e)}"
            }
    
    def evaluate_sql(self, query: str, sql: str, schema: str) -> Dict:
        """
        Evaluate SQL using TruLens
        
        Args:
            query: Natural language query
            sql: Generated SQL
            schema: Database schema as string
            
        Returns:
            Dict containing evaluation results
        """
        # Clean SQL before evaluation
        sql = self.clean_sql(sql)
        
        # Nếu TruLens không khả dụng, trả về cấu trúc giả lập với thông báo
        if not self.trulens_available:
            return {
                "success": True,
                "message": "TruLens evaluation skipped (not available)",
                "all_passed": True,
                "hallucination_score": 0.0,
                "relevancy_score": 0.0,
                "contextual_relevancy_score": 0.0,
                "correctness_score": 0.0,
                "syntax_score": 0.0,
                "metrics_available": False,
                "reason": "OpenAI API key not set or TruLens not installed",
                "evaluation_framework": "TruLens (simulated)"
            }
        
        try:
            # Đánh giá relevance
            relevance_score = self.relevance.score(
                query=query, response=sql
            )
            
            # Đánh giá groundedness (hallucination)
            groundedness_score = self.groundedness.score(
                response=sql, context=[schema]
            )
            
            # Đánh giá context relevance
            context_relevance_score = self.context_relevance.score(
                query=query, response=sql, context=[schema]
            )
            
            # Đánh giá SQL correctness
            correctness_score = self.sql_correctness.score(
                question=query, schema=schema, sql=sql
            )
            
            # Đánh giá SQL syntax
            syntax_score = self.sql_syntax.score(
                sql=sql
            )
            
            # Tính toán all_passed dựa trên ngưỡng
            all_passed = (
                relevance_score >= self.thresholds["relevance"] and
                groundedness_score >= self.thresholds["groundedness"] and
                context_relevance_score >= self.thresholds["context_relevance"] and
                correctness_score >= self.thresholds["correctness"] and
                syntax_score >= self.thresholds["syntax"]
            )
            
            # Trả về kết quả đánh giá
            return {
                "success": True,
                "all_passed": all_passed,
                "metrics_available": True,
                "relevancy_score": relevance_score,
                "hallucination_score": 1.0 - groundedness_score,  # Đảo ngược để phù hợp với DeepEval
                "contextual_relevancy_score": context_relevance_score,
                "correctness_score": correctness_score,
                "syntax_score": syntax_score,
                "evaluation_framework": "TruLens"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"TruLens error: {error_msg}")
            
            # Kiểm tra lỗi quyền truy cập mô hình
            if "does not have access to model" in error_msg:
                return {
                    "success": False,
                    "error": "OpenAI API key valid but no access to required model",
                    "all_passed": False,
                    "metrics_available": False,
                    "reason": f"Your OpenAI account does not have access to the required model. Try using a different model.",
                    "evaluation_framework": "TruLens (error)"
                }
            
            return {
                "success": False,
                "error": f"Error evaluating SQL: {error_msg}",
                "all_passed": False,
                "metrics_available": False,
                "reason": error_msg,
                "evaluation_framework": "TruLens (error)"
            }
    
    def process_query(self, query: str, schema: str) -> Dict:
        """
        Process a natural language query to SQL
        
        Args:
            query: Natural language query
            schema: Database schema as string
            
        Returns:
            Dict containing processing results
        """
        result = {
            "original_query": query,
            "processed_query": query,
            "source": "generated",
            "is_ambiguous": False,
            "clarification_needed": False,
            "sql": "",
            "explanation": "",
            "evaluation": None
        }
        
        # Step 1: Check FAQ
        faq_match = self.match_faq(query)
        if faq_match:
            result["source"] = "faq"
            result["sql"] = self.clean_sql(faq_match.get("sql", ""))
            result["explanation"] = faq_match.get("explanation", "")
            # Skip ambiguity detection and evaluation for FAQ matches
            return result
        
        # Step 2: Detect ambiguity
        is_ambiguous, ambiguity_analysis = self.detect_ambiguity(query, schema)
        if is_ambiguous:
            result["is_ambiguous"] = True
            result["ambiguity_analysis"] = ambiguity_analysis
            result["clarification_questions"] = self.generate_clarification_questions(query, ambiguity_analysis, schema)
            result["clarification_needed"] = True
            return result
        
        # Step 3: Generate SQL
        sql_result = self.generate_sql(query, schema)
        if not sql_result.get("success", False):
            result["error"] = sql_result.get("error", "Unknown error generating SQL")
            return result
        
        result["sql"] = sql_result.get("sql", "")
        result["explanation"] = sql_result.get("explanation", "")
        
        # Step 4: Evaluate SQL
        if result["sql"]:
            evaluation = self.evaluate_sql(query, result["sql"], schema)
            result["evaluation"] = evaluation
        
        return result
    
    def process_clarified_query(self, original_query: str, clarification_responses: Dict[str, str], schema: str) -> Dict:
        """
        Process a query after clarification
        
        Args:
            original_query: Original natural language query
            clarification_responses: Dict of clarification responses
            schema: Database schema as string
            
        Returns:
            Dict containing processing results
        """
        # Update the query based on clarification responses
        updated_query = self.update_query_with_clarification(original_query, clarification_responses)
        
        result = {
            "original_query": original_query,
            "processed_query": updated_query,
            "source": "clarified",
            "is_ambiguous": False,
            "clarification_needed": False,
            "clarification_responses": clarification_responses,
            "sql": "",
            "explanation": "",
            "evaluation": None
        }
        
        # Generate SQL for the updated query
        sql_result = self.generate_sql(updated_query, schema)
        if not sql_result.get("success", False):
            result["error"] = sql_result.get("error", "Unknown error generating SQL")
            return result
        
        result["sql"] = sql_result.get("sql", "")
        result["explanation"] = sql_result.get("explanation", "")
        
        # Evaluate SQL
        if result["sql"]:
            evaluation = self.evaluate_sql(updated_query, result["sql"], schema)
            result["evaluation"] = evaluation
        
        return result
