import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf as fitz
import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_qr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --- CẤU HÌNH KẾT NỐI POSTGRESQL ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "postgresql://postgres:ALTcgCCqAIjJCAwtcSVOFqdsBbNoBzyq@postgres.railway.internal:5432/railway"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "API is running successfully!"}

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        ocr_database = {}

        # 1. TẢI TRƯỚC TOÀN BỘ DANH MỤC HÀNG HÓA TỪ BẢNG "danh muc hang trai 2"
        master_items = []
        try:
            sql_items = text('SELECT "Mã hàng", "Tên hàng", "Đơn vị tính" FROM "danh muc hang trai 2";')
            result_items = db.execute(sql_items).fetchall()
            for row in result_items:
                master_items.append({
                    "ma": str(row[0] or ""),
                    "ten": str(row[1] or ""),
                    "dvt": str(row[2] or ""),
                    "sl": 0,
                    "trangThai": "Khớp DB"
                })
        except Exception as e:
            print(f"Lỗi đọc bảng 'danh muc hang trai 2': {str(e)}")

        # Nếu bảng danh mục trống, dùng dữ liệu mẫu để giao diện không bị trắng
        if not master_items:
            master_items = [
                {"ma": "ERR", "ten": "Không thể đọc dữ liệu từ bảng 'danh muc hang trai 2'", "dvt": "--", "sl": 0, "trangThai": "Lỗi DB"}
            ]

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue

            # 2. ĐỌC MÃ QR CODE (TÌM THÔNG TIN TỪ BẢNG "_A2" THEO "STT")
            qr_codes = decode_qr(img)
            ten_khach = "Không tìm thấy mã QR trên trang"
            tien_con = "0 đ"
            
            if qr_codes:
                qr_data = qr_codes[0].data.decode('utf-8').strip()
                try:
                    sql_a2 = text('SELECT * FROM "_A2" WHERE "STT" = :stt LIMIT 1;')
                    a2_result = db.execute(sql_a2, {"stt": qr_data}).fetchone()
                    
                    if a2_result:
                        # Chuyển kết quả sang dạng dict để dễ bóc tách tên cột không phân biệt hoa thường
                        row_dict = a2_result._asdict()
                        # Tìm kiếm linh hoạt các cột tên khách hàng / số tiền trong bảng _A2
                        found_name = None
                        found_money = None
                        for k, v in row_dict.items():
                            k_lower = k.lower()
                            if any(w in k_lower for w in ["ten", "họ tên", "hoten", "phamnhan"]):
                                found_name = v
                            if any(w in k_lower for w in ["tien", "số tiền", "sotien", "con"]):
                                found_money = v
                        
                        ten_khach = str(found_name or f"STT: {qr_data} (Thiếu cột tên)")
                        tien_con = str(found_money or "0") + " đ"
                    else:
                        ten_khach = f"Mã QR [{qr_data}] không có trong bảng _A2"
                except Exception as db_err:
                    print(f"Lỗi truy vấn bảng _A2: {str(db_err)}")
                    ten_khach = f"Mã QR [{qr_data}] (Lỗi SQL _A2)"

            # Gán danh mục hàng hóa vào từng trang
            ocr_database[page_num + 1] = {
                "ten": ten_khach,
                "tienCon": tien_con,
                "items": master_items
            }

        return {"total_pages": len(doc), "data": ocr_database}
        
    except Exception as e:
        print("LỖI XỬ LÝ PDF:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

class PageDataModel(BaseModel):
    page: int
    ten: str
    tienCon: str
    items: list

@app.post("/api/save-page")
async def save_page(data: PageDataModel):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM phieu_mua_hang WHERE page = :page"), {"page": data.page})
        
        for item in data.items:
            insert_sql = text("""
                INSERT INTO phieu_mua_hang (page, ten_pham_nhan, tien_con, ma_hang, ten_hang, dvt, so_luong, trang_thai)
                VALUES (:page, :ten, :tien_con, :ma_hang, :ten_hang, :dvt, :so_luong, :trang_thai)
            """)
            db.execute(insert_sql, {
                "page": data.page,
                "ten": data.ten,
                "tien_con": data.tienCon,
                "ma_hang": item["ma"],
                "ten_hang": item["ten"],
                "dvt": item["dvt"],
                "so_luong": item["sl"],
                "trang_thai": "Đã duyệt"
            })
            
        db.commit()
        return {"status": "success", "message": f"Đã lưu thành công dữ liệu trang {data.page}!"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
