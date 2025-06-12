# Text2SQL Demo

Ứng dụng demo chuyển đổi câu hỏi tiếng Việt thành câu truy vấn SQL, sử dụng mô hình ngôn ngữ lớn (LLM) và TruLens để đánh giá độ tin cậy.

## Tính năng chính

- Chuyển đổi câu hỏi tiếng Việt thành câu truy vấn SQL
- Hỗ trợ tải lên file schema và dữ liệu mẫu
- Hiển thị kết quả truy vấn dưới dạng bảng
- Đánh giá độ tin cậy của câu truy vấn bằng TruLens
- Giao diện người dùng thân thiện với Bootstrap 5

## Yêu cầu hệ thống

- Python 3.8+
- Flask
- OpenAI API key
- TruLens
- Các thư viện Python khác (xem requirements.txt)

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd Text2SQL
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

3. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

4. Thiết lập biến môi trường:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Cấu trúc thư mục
Text2SQL/
├── src/
│ ├── static/
│ │ └── index_trulens.html # Giao diện người dùng
│ ├── routes/
│ │ └── api_trulens.py # API endpoints
│ └── utils/
│ └── trulens_utils.py # Tiện ích TruLens
├── grocery_sales_schema.json # Schema mẫu
├── default.sqlite # Database mẫu
├── requirements.txt
└── README.md

## Sử dụng

1. Khởi động server Flask:
```bash
python src/app.py
```

2. Mở trình duyệt và truy cập:
```
http://localhost:5000
```

3. Tải lên file schema và dữ liệu mẫu (nếu cần)

4. Nhập câu hỏi tiếng Việt vào ô input

5. Nhấn nút "Gửi" hoặc Enter để thực hiện truy vấn

## Cách hoạt động

1. Người dùng nhập câu hỏi tiếng Việt
2. Hệ thống chuyển đổi câu hỏi thành câu truy vấn SQL
3. TruLens đánh giá độ tin cậy của câu truy vấn
4. Kết quả được hiển thị dưới dạng bảng
5. Người dùng có thể xem chi tiết đánh giá của TruLens

## Ví dụ câu hỏi

- "Chi nhánh nào đang bán tốt nhất tính theo tiêu chí quantity_sold?"
- "Top 5 sản phẩm bán chạy nhất trong tháng 1/2024"
- "Tổng doanh thu theo từng chi nhánh"

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request để đóng góp.

## Giấy phép

[MIT License](LICENSE)

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào, vui lòng tạo issue trong repository.

