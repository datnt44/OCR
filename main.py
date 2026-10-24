import os
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymupdf as fitz
import pytesseract
import cv2
import numpy as np
from pyzbar.pyzbar import decode as decode_qr
from sqlalchemy import create_engine, text
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
    return {"status": "online", "message": "OCR & Database Integration API is running!"}

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    db = SessionLocal()
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

            # --- 1. ĐỌC MÃ QR CODE (CHÍNH LÀ CỘT "STT" TRONG BẢNG _A2) ---
            qr_codes = decode_qr(img)
            buyer_info = {"ten": "Không quét được mã QR", "tienCon": "0 đ"}
            
            if qr_codes:
                qr_data = qr_codes[0].data.decode('utf-8').strip()
                try:
                    # Truy vấn bảng _A2 với khóa chính là cột "STT"
                    # Lưu ý: Bọc tên cột bằng dấu ngoặc kép hoặc đổi tên chữ hoa/thường tùy theo định nghĩa trong DB
                    sql_a2 = text('SELECT * FROM "_A2" WHERE "STT" = :stt LIMIT 1;')
                    result = db.execute(sql_a2, {"stt": qr_data}).fetchone()
                    
                    if result:
                        # Lấy thông tin người mua (điều chỉnh tên cột sao cho khớp với bảng _A2 của bạn, ví dụ: "ho_ten", "ten_pham_nhan"...)
                        # Dưới đây tạm lấy các cột phổ biến hoặc trả về dạng đối tượng
                        ten_khach = getattr(result, "ten_pham_nhan", getattr(result, "HoTen", f"STT: {qr_data}"))
                        tien_con = getattr(result, "so_tien_con", getattr(result, "TienCon", "0"))
                        
                        buyer_info["ten"] = str(ten_khach)
                        buyer_info["tienCon"] = str(tien_con) + " đ"
                    else:
                        buyer_info["ten"] = f"Mã QR [{qr_data}] không tồn tại trong bảng _A2"
                except Exception as db_err:
                    print(f"Lỗi truy vấn bảng _A2: {str(db_err)}")
                    buyer_info["ten"] = f"Mã QR [{qr_data}] (Lỗi kết nối DB _A2)"

            # --- 2. LẤY DỮ LIỆU TỪ BẢNG "danh muc hang trai 2" ---
            page_items = []
            try:
                # Truy vấn lấy Mã hàng, Tên hàng, Đơn vị tính từ bảng danh mục
                sql_items = text('SELECT "Mã hàng", "Tên hàng", "Đơn vị tính" FROM "danh muc hang trai 2";')
                items_result = db.execute(sql_items).fetchall()
                
                if items_result:
                    for row in items_result:
                        page_items.append({
                            "ma": row[0],         # "Mã hàng"
                            "ten": row[1],        # "Tên hàng"
                            "dvt": row[2],        # "Đơn vị tính"
                            "sl": 0,              # Số lượng mặc định để rà soát
                            "trangThai": "Đã lấy từ danh mục"
                        })
                else:
                    page_items = [{"ma": "TRONG", "ten": "Bảng danh mục trống", "dvt": "--", "sl": 0, "trangThai": "Cảnh báo"}]
            except Exception as e:
                print(f"Lỗi lấy bảng danh mục hàng hóa: {str(e)}")
                page_items = [{"ma": "ERR", "ten": "Lỗi kết nối bảng danh muc hang trai 2", "dvt": "--", "sl": 0, "trangThai": "Lỗi DB"}]

            ocr_database[page_num + 1] = {
                "ten": buyer_info["ten"],
                "tienCon": buyer_info["tienCon"],
                "items": page_items
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
        # Xóa dữ liệu cũ của trang trước khi lưu mới
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
        return {"status": "success", "message": f"Đã lưu thành công dữ liệu trang {data.page} vào cơ sở dữ liệu!"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
