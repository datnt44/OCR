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
    return {"status": "online", "message": "FastAPI PDF & Database Backend is running successfully!"}

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        ocr_database = {}

        # 1. TẢI TOÀN BỘ DANH MỤC HÀNG HÓA TỪ BẢNG "danh muc hang trai 2"
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
            db.rollback()
            error_msg = str(e)
            print(f"LỖI KHI ĐỌC BẢNG DANH MỤC: {error_msg}")
            master_items = [{
                "ma": "ERR", 
                "ten": f"Lỗi bảng danh mục: {error_msg[:60]}", 
                "dvt": "--", 
                "sl": 0, 
                "trangThai": "Lỗi DB"
            }]

        if not master_items:
            master_items = [{
                "ma": "TRONG", 
                "ten": "Bảng 'danh muc hang trai 2' không có dữ liệu", 
                "dvt": "--", 
                "sl": 0, 
                "trangThai": "Trống"
            }]

        # 2. XỬ LÝ TỪNG TRANG TRONG FILE PDF
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue

            # ĐỌC MÃ QR CODE (CHÍNH LÀ "STT" TRONG BẢNG _A2 ĐỂ LẤY "Họ và tên")
            qr_codes = decode_qr(img)
            ten_khach = "Không tìm thấy mã QR trên trang"
            tien_con = "0 đ"
            
            if qr_codes:
                qr_data = qr_codes[0].data.decode('utf-8').strip()
                try:
                    # Truy vấn bảng _A2 với điều kiện "STT" khớp mã QR và lấy cột "Họ và tên"
                    sql_a2 = text('SELECT "Họ và tên", "Số tiền" FROM "_A2" WHERE "STT"::text = :stt LIMIT 1;')
                    a2_result = db.execute(sql_a2, {"stt": qr_data}).fetchone()
                    
                    if a2_result:
                        ho_ten = a2_result[0]
                        so_tien = a2_result[1] if len(a2_result) > 1 else 0
                        
                        ten_khach = str(ho_ten or f"STT: {qr_data} (Tên trống)")
                        tien_con = str(so_tien or "0") + " đ"
                    else:
                        ten_khach = f"Mã QR [{qr_data}] không tồn tại trong bảng _A2"
                except Exception as db_err:
                    db.rollback() 
                    try:
                        sql_fallback = text('SELECT * FROM "_A2" WHERE "STT"::text = :stt LIMIT 1;')
                        row_fb = db.execute(sql_fallback, {"stt": qr_data}).fetchone()
                        if row_fb:
                            row_dict = row_fb._asdict()
                            name_val = None
                            money_val = None
                            for k, v in row_dict.items():
                                k_l = k.lower()
                                if "tên" in k_l or "ten" in k_l:
                                    name_val = v
                                if "tiền" in k_l or "tien" in k_l or "con" in k_l or "sodu" in k_l:
                                    money_val = v
                            ten_khach = str(name_val or f"STT: {qr_data}")
                            tien_con = str(money_val or "0") + " đ"
                        else:
                            ten_khach = f"Mã QR [{qr_data}] không tìm thấy trong _A2"
                    except Exception as e2:
                        db.rollback()
                        err_str = str(e2)
                        print(f"LỖI TRUY VẤN _A2: {err_str}")
                        ten_khach = f"Lỗi ánh xạ STT: {err_str[:40]}"

            ocr_database[page_num + 1] = {
                "ten": ten_khach,
                "tienCon": tien_con,
                "items": master_items
            }

        return {"total_pages": len(doc), "data": ocr_database}
        
    except Exception as e:
        db.rollback()
        print("LỖI XỬ LÝ CHUNG:", traceback.format_exc())
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
        return {"status": "success", "message": f"Đã lưu thành công dữ liệu trang {data.page} vào cơ sở dữ liệu!"}
    except Exception as e:
        db.rollback()
        print("LỖI KHI LƯU TRANG:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
