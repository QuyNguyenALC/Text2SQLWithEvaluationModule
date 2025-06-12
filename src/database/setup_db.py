import sqlite3
import pandas as pd
import os

def setup_database():
    # Tạo thư mục database nếu chưa tồn tại
    os.makedirs('src/database', exist_ok=True)
    
    # Kết nối đến database
    conn = sqlite3.connect('src/database/grocery.db')
    
    try:
        # Đọc file CSV và tự động tạo bảng
        df = pd.read_csv('/Users/ap24h/Documents/Workfolder/KiotViet/Prj/Text2SQL/src/database/grocery_sales_data_fixed.csv')
        
        # Tự động tạo bảng và import dữ liệu
        df.to_sql('grocery_sales', conn, if_exists='replace', index=False)
        
        print("Database setup completed successfully!")
        print(f"Imported {len(df)} records into grocery_sales table")
        
        # In ra cấu trúc bảng để kiểm tra
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(grocery_sales)")
        columns = cursor.fetchall()
        print("\nTable structure:")
        for col in columns:
            print(f"- {col[1]} ({col[2]})")
            
    except FileNotFoundError:
        print("Error: grocery_sales_data.csv not found!")
        print("Please make sure the file exists in the correct directory")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    setup_database() 