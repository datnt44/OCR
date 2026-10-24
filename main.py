import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Cấu hình CORS chuẩn để chấp nhận mọi request từ Netlify (bao gồm cả Preflight OPTIONS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gọi vào
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

from pydantic import BaseModel
import pymupdf as fitz  # PyMuPDF chuẩn
import pytesseract
import cv2
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- CẤU HÌNH KẾT NỐI POSTGRESQL ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgres:ALTcgCCqAIjJCAwtcSVOFqdsBbNoBzyq@shinkansen.proxy.rlwy.net:57724/railway"

# Ép chuỗi kết nối sử dụng đúng driver psycopg2 để tránh lỗi psycopg3
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Định nghĩa bảng cơ sở dữ liệu trong PostgreSQL
class PhieuMuaHang(Base):
    __tablename__ = "phieu_mua_hang"
    id = Column(Integer, primary_key=True, index=True)
    page = Column(Integer)
    ten_pham_nhan = Column(String(255))
    tien_con = Column(String(50))
    ma_hang = Column(String(50))
    ten_hang = Column(Text)
    dvt = Column(String(50))
    so_luong = Column(Integer)
    trang_thai = Column(String(50))

# Tự động tạo bảng nếu chưa tồn tại
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Bật CORS đầy đủ để Netlify gọi sang không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Danh sách mặt hàng mẫu để đối chiếu OCR
STANDARD_ITEMS = [
    {"ma": "A26100001", "ten": "Nước tinh khiết Number 1 500ml", "dvt": "Chai"},
    {"ma": "A26100002", "ten": "Bánh bông lan Gibon 156g", "dvt": "Gói"},
    {"ma": "A26100003", "ten": "Lương khô 5+", "dvt": "Gói"},
    {"ma": "A26100005", "ten": "Lương khô dinh dưỡng 350g", "dvt": "Gói"},
    {"ma": "A26100006", "ten": "Cam", "dvt": "Kg"},
    {"ma": "A26100009", "ten": "Sữa đậu nành Fami nguyên chất 200ml", "dvt": "Hộp"},
    {"ma": "A26100021", "ten": "Giò lụa 0,4kg", "dvt": "Túi"},
    {"ma": "A26100022", "ten": "Chả lợn 0,4kg", "dvt": "Túi"},
    {"ma": "A26100024", "ten": "Ruốc thịt 0,2kg", "dvt": "Túi"}
]

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        ocr_database = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Chạy Tesseract OCR tiếng Việt
            text = pytesseract.image_to_string(img, lang='vie')
            
            ten_khach = "Không rõ"
            tien_con = "0 đ"
            for line in text.split('\n'):
                if "Họ và tên:" in line:
                    ten_khach = line.split("Họ và tên:")[-1].strip()
                if "Tiền còn:" in line or "Tiên còn:" in line:
                    tien_con = line.split(":")[-1].strip() + " đ"

            # Đối chiếu mặt hàng
            page_items = []
            for item in STANDARD_ITEMS:
                is_detected = item["ten"].lower() in text.lower()
                page_items.append({
                    "ma": item["ma"],
                    "ten": item["ten"],
                    "dvt": item["dvt"],
                    "sl": 1 if is_detected else 0,
                    "trangThai": "Khớp" if is_detected else "Nghi ngờ"
                })
            
            filtered_items = [i for i in page_items if i["sl"] > 0]

            ocr_database[page_num + 1] = {
                "ten": ten_khach,
                "tienCon": tien_con,
                "items": filtered_items if filtered_items else page_items[:3]
            }

        return {"total_pages": len(doc), "data": ocr_database}
        
    except Exception as e:
        print("LỖI OCR CHI TIẾT:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class PageDataModel(BaseModel):
    page: int
    ten: str
    tienCon: str
    items: list

@app.post("/api/save-page")
async def save_page(data: PageDataModel):
    db = SessionLocal()
    try:
        # Xóa dữ liệu cũ của trang này trước khi ghi mới để tránh trùng lặp
        db.query(PhieuMuaHang).filter(PhieuMuaHang.page == data.page).delete()
        
        for item in data.items:
            db_item = PhieuMuaHang(
                page=data.page,
                ten_pham_nhan=data.ten,
                tien_con=data.tienCon,
                ma_hang=item["ma"],
                ten_hang=item["ten"],
                dvt=item["dvt"],
                so_luong=item["sl"],
                trang_thai="Đã duyệt"
            )
            db.add(db_item)
            
        db.commit()
        return {"status": "success", "message": f"Đã lưu thành công dữ liệu trang {data.page} vào PostgreSQL!"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    import os
    # Lấy cổng PORT do Railway tự động cung cấp, nếu chạy local thì mặc định là 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
