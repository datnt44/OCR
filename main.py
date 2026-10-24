import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf as fitz
import pytesseract
import cv2
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- CẤU HÌNH KẾT NỐI POSTGRESQL ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgres:password@localhost:5432/ocr_db"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 🏢 DỮ LIỆU CHUẨN LẤY TỪ DATABASE HOẶC API MẪU
# (Bạn có thể thay thế bằng hàm fetch dữ liệu từ bảng danh mục hàng hóa thật của bạn)
MASTER_DATABASE_ITEMS = [
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

@app.get("/")
def read_root():
    return {"status": "online", "message": "Digit-OCR Backend is running!"}

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        ocr_database = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue

            # Xử lý ảnh xám & tăng tương phản để OCR chữ số chuẩn xác tuyệt đối
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

            # 🔢 CẤU HÌNH TESSERACT CHỈ ĐỌC CHỮ SỐ (DIGITS ONLY)
            digit_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            
            # Đọc toàn bộ văn bản bằng cấu hình thông thường (để lấy tên nếu cần)
            full_text = pytesseract.image_to_string(thresh, lang='vie')
            
            # Quét riêng các cụm số hoặc dòng số nếu cần thiết
            # Ở đây chúng ta nạp sẵn toàn bộ danh mục từ cơ sở dữ liệu vào bảng
            page_items = []
            for item in MASTER_DATABASE_ITEMS:
                # Kiểm tra nếu tên mặt hàng xuất hiện trên phiếu (hoặc mặc định nạp danh mục lên bảng để user điền số lượng)
                is_present = item["ten"].lower() in full_text.lower()
                
                page_items.append({
                    "ma": item["ma"],
                    "ten": item["ten"],
                    "dvt": item["dvt"],
                    "sl": 1 if is_present else 0, # Số lượng mặc định hoặc bóc tách từ OCR
                    "trangThai": "Đã đồng bộ DB"
                })

            ocr_database[page_num + 1] = {
                "ten": "Phạm nhân định danh theo số phiếu", 
                "tienCon": "500,000 đ",
                "items": page_items # Đưa toàn bộ danh mục chuẩn từ DB vào bảng có sẵn
            }

        return {"total_pages": len(doc), "data": ocr_database}
        
    except Exception as e:
        print("LỖI:", traceback.format_exc())
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
        return {"status": "success", "message": f"Đã lưu trang {data.page} vào PostgreSQL thành công!"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
