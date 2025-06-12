import sqlite3
import os

def check_database():
    # Đường dẫn đến database
    db_path = os.path.join(os.path.dirname(__file__), 'grocery.db')
    
    print(f"Checking database at: {db_path}")
    print(f"Database file exists: {os.path.exists(db_path)}")
    
    if os.path.exists(db_path):
        try:
            # Kết nối đến database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Kiểm tra bảng grocery_sales
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='grocery_sales'")
            table_exists = cursor.fetchone() is not None
            print(f"Table grocery_sales exists: {table_exists}")
            
            if table_exists:
                # Đếm số lượng bản ghi
                cursor.execute("SELECT COUNT(*) FROM grocery_sales")
                count = cursor.fetchone()[0]
                print(f"Number of records in grocery_sales: {count}")
                
                # Lấy 5 bản ghi đầu tiên
                cursor.execute("SELECT * FROM grocery_sales LIMIT 5")
                rows = cursor.fetchall()
                print("\nFirst 5 records:")
                for row in rows:
                    print(row)
            
        except Exception as e:
            print(f"Error checking database: {str(e)}")
        finally:
            conn.close()
    else:
        print("Database file not found!")

if __name__ == "__main__":
    check_database() 