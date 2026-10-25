import copy
import difflib
import logging
import os
import re
import traceback
import unicodedata
import uuid
from typing import List

import cv2
import numpy as np
import pymupdf as fitz
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyzbar.pyzbar import decode as decode_qr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("phieu-api")

# ======================================================================
# CẤU HÌNH (tất cả đọc từ biến môi trường)
# ======================================================================
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Thiếu biến môi trường DATABASE_URL")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Danh sách domain frontend, ngăn cách bằng dấu phẩy. Ví dụ:
# ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

PDF_DPI = int(os.environ.get("PDF_DPI", "200"))          # độ phân giải render trang
QR_FALLBACK_DPI = int(os.environ.get("QR_FALLBACK_DPI", "300"))
MAX_PDF_MB = int(os.environ.get("MAX_PDF_MB", "30"))
OCR_LANG_WANTED = os.environ.get("OCR_LANG", "vie+eng")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    log.warning("ALLOWED_ORIGINS chưa được đặt -> đang cho phép mọi origin (không kèm credentials).")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


def _pick_ocr_lang() -> str:
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    chosen = [l for l in OCR_LANG_WANTED.split("+") if l in available]
    return "+".join(chosen) or "eng"


OCR_LANG = _pick_ocr_lang()
log.info("Ngôn ngữ OCR: %s", OCR_LANG)


@app.on_event("startup")
def ensure_schema():
    """Thêm cột batch_id vào bảng phieu_mua_hang nếu chưa có."""
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE phieu_mua_hang ADD COLUMN IF NOT EXISTS batch_id TEXT"))
        db.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_phieu_batch_page ON phieu_mua_hang (batch_id, page)"
        ))
        db.commit()
    except Exception:
        db.rollback()
        log.warning("Không thể cập nhật schema phieu_mua_hang:\n%s", traceback.format_exc())
    finally:
        db.close()


# ======================================================================
# TIỆN ÍCH CHUỖI
# ======================================================================
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def norm_code(s: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", s.upper())


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ======================================================================
# ĐỌC QR
# ======================================================================
def read_qr(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    for candidate in (gray, otsu):
        codes = decode_qr(candidate)
        if codes:
            return codes[0].data.decode("utf-8", errors="ignore").strip()
    try:
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        if data:
            return data.strip()
    except cv2.error:
        pass
    return None


def render_page(page, dpi: int) -> np.ndarray | None:
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.tobytes("png"), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ======================================================================
# OCR BẢNG: tìm cột "Mã hàng" và "Số lượng"
# ======================================================================
def ocr_words(img: np.ndarray) -> list:
    data = pytesseract.image_to_data(
        img, lang=OCR_LANG, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        t = raw.strip()
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({
            "text": t, "n": strip_accents(t),
            "x": x, "y": y, "w": w, "h": h,
            "cx": x + w / 2, "cy": y + h / 2,
        })
    return words


def find_header_pair(words: list, first: str, second: str):
    """Tìm cặp từ liền nhau trên cùng một dòng, ví dụ ('ma','hang') hoặc ('so','luong')."""
    best = None
    for a in words:
        if sim(a["n"], first) < 0.75:
            continue
        for b in words:
            if b is a or sim(b["n"], second) < 0.75:
                continue
            if abs(a["cy"] - b["cy"]) > max(a["h"], b["h"]):
                continue
            gap = b["x"] - (a["x"] + a["w"])
            if gap < -a["h"] or gap > a["h"] * 4:
                continue
            if best is None or a["cy"] < best[0]["cy"]:
                best = (a, b)
    return best


def column_bounds(words: list, a: dict, b: dict, img_w: int):
    """Ranh giới cột = điểm giữa khoảng trống với tiêu đề cột kế bên."""
    cy = (a["cy"] + b["cy"]) / 2
    tol = max(a["h"], b["h"])
    row = sorted((w for w in words if abs(w["cy"] - cy) <= tol), key=lambda w: w["x"])
    left_edge = a["x"]
    right_edge = b["x"] + b["w"]
    prev = [w for w in row if w["x"] + w["w"] <= left_edge]
    nxt = [w for w in row if w["x"] >= right_edge]
    lo = (prev[-1]["x"] + prev[-1]["w"] + left_edge) / 2 if prev else 0
    hi = (right_edge + nxt[0]["x"]) / 2 if nxt else img_w
    return int(lo), int(hi)


def read_digits(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> str:
    """OCR một ô, chỉ nhận các chữ số 0-9."""
    inset = 6  # cắt bớt viền ô để đường kẻ không bị đọc thành số "1"
    x0, x1 = max(x0 + inset, 0), max(x1 - inset, 0)
    y0, y1 = max(y0, 0), max(y1, 0)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(bw, lang="eng", config=cfg)
    return re.sub(r"\D", "", txt)


def extract_table_rows(img: np.ndarray):
    """Trả về (danh sách (mã hàng, số lượng), danh sách cảnh báo)."""
    warnings = []
    words = ocr_words(img)
    ma_pair = find_header_pair(words, "ma", "hang")
    sl_pair = find_header_pair(words, "so", "luong")
    if not ma_pair or not sl_pair:
        missing = []
        if not ma_pair:
            missing.append("Mã hàng")
        if not sl_pair:
            missing.append("Số lượng")
        return [], [f"Không tìm thấy tiêu đề cột: {', '.join(missing)}"]

    img_h, img_w = img.shape[:2]
    ma_lo, ma_hi = column_bounds(words, *ma_pair, img_w)
    sl_lo, sl_hi = column_bounds(words, *sl_pair, img_w)

    header_bottom = max(w["y"] + w["h"] for w in (*ma_pair, *sl_pair))
    body = sorted((w for w in words if w["cy"] > header_bottom), key=lambda w: w["cy"])
    if not body:
        return [], ["Không có dòng dữ liệu bên dưới tiêu đề"]

    med_h = float(np.median([w["h"] for w in body])) or 20.0

    rows = []
    for w in body:
        if rows and abs(w["cy"] - rows[-1]["cy"]) <= med_h * 0.7:
            rows[-1]["words"].append(w)
            rows[-1]["cy"] = float(np.mean([x["cy"] for x in rows[-1]["words"]]))
        else:
            rows.append({"cy": w["cy"], "words": [w]})

    pad = int(med_h * 0.3)
    result = []
    for row in rows:
        ma_words = sorted(
            (w for w in row["words"] if ma_lo <= w["cx"] <= ma_hi), key=lambda w: w["x"]
        )
        if not ma_words:
            continue
        code = re.sub(r"[^0-9A-Za-z\-_./]", "", "".join(w["text"] for w in ma_words)).upper()
        if len(code) < 2:
            continue
        y0 = min(w["y"] for w in row["words"]) - pad
        y1 = max(w["y"] + w["h"] for w in row["words"]) + pad
        digits = read_digits(img, sl_lo, sl_hi, y0, min(y1, img_h))
        result.append((code, int(digits) if digits else 0))
    if not result:
        warnings.append("Tìm thấy tiêu đề nhưng không đọc được dòng hàng hóa nào")
    return result, warnings


# ======================================================================
# TRUY VẤN DATABASE
# ======================================================================
MASTER_TABLE = "danh muc hang trai 2"


def _find_column(columns: list, *candidates: str):
    """Tìm tên cột thật, bỏ qua dấu / hoa thường / khoảng trắng."""
    def key(s: str) -> str:
        return strip_accents(s).replace(" ", "").replace("_", "")

    wanted = {key(c) for c in candidates}
    for col in columns:
        if key(col) in wanted:
            return col
    return None


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def load_master_items(db) -> list:
    try:
        columns = [
            r[0]
            for r in db.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :t ORDER BY ordinal_position"
                ),
                {"t": MASTER_TABLE},
            ).fetchall()
        ]
        log.info("Các cột trong bảng %s: %s", MASTER_TABLE, columns)

        col_ma = _find_column(columns, "Mã hàng", "Mã số hàng", "Mã")
        col_ten = _find_column(columns, "Tên hàng", "Tên hàng hóa", "Tên")
        col_dvt = _find_column(columns, "Đơn vị tính", "Đơn vị", "ĐVT", "DVT")

        if not columns or not col_ma or not col_ten:
            log.error("Bảng %s thiếu cột Mã hàng / Tên hàng. Cột hiện có: %s", MASTER_TABLE, columns)
            raise HTTPException(
                status_code=500,
                detail="Bảng danh mục không có cột Mã hàng hoặc Tên hàng (xem log máy chủ)",
            )
        if not col_dvt:
            log.warning("Bảng %s không có cột đơn vị tính, sẽ để trống.", MASTER_TABLE)

        dvt_sql = _quote(col_dvt) if col_dvt else "''"
        rows = db.execute(
            text(f"SELECT {_quote(col_ma)}, {_quote(col_ten)}, {dvt_sql} FROM {_quote(MASTER_TABLE)}")
        ).fetchall()
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        log.error("Lỗi đọc bảng danh mục:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Không đọc được bảng danh mục hàng hóa")
    return [
        {
            "ma": str(r[0] or ""),
            "ten": str(r[1] or ""),
            "dvt": str(r[2] or ""),
            "sl": 0,
            "trangThai": "Chưa đọc được",
        }
        for r in rows
    ]


def lookup_customer(db, stt: str) -> str:
    try:
        row = db.execute(
            text('SELECT "Họ và tên" FROM "_A2" WHERE "STT"::text = :stt LIMIT 1;'),
            {"stt": stt},
        ).fetchone()
    except Exception:
        db.rollback()
        log.error("Lỗi truy vấn _A2 (STT=%s):\n%s", stt, traceback.format_exc())
        return "Lỗi truy vấn bảng _A2"
    if row and row[0]:
        return str(row[0])
    return f"Mã QR [{stt}] không tồn tại trong bảng _A2"


# Các ký tự OCR hay nhầm với chữ số, dùng khi so khớp mã hàng
_CONFUSION = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})


def _confuse(key: str) -> str:
    return key.translate(_CONFUSION)


def merge_ocr_into_items(master_items: list, ocr_rows: list) -> list:
    """Chỉ trả các mã hàng OCR được trên phiếu, ánh xạ sang bảng danh mục để lấy tên/ĐVT."""
    by_code, by_conf = {}, {}
    for it in master_items:
        if not it["ma"]:
            continue
        k = norm_code(it["ma"])
        by_code[k] = it
        by_conf.setdefault(_confuse(k), it)
    keys = list(by_code.keys())

    items, seen = [], {}
    for code, qty in ocr_rows:
        key = norm_code(code)
        master, status = by_code.get(key), "Khớp DB"
        if master is None:
            master, status = by_conf.get(_confuse(key)), "Khớp gần đúng"
        if master is None and keys:
            close = difflib.get_close_matches(key, keys, n=1, cutoff=0.85)
            if close:
                master, status = by_code[close[0]], "Khớp gần đúng"

        if master is not None:
            uid = master["ma"]
            if uid in seen:
                seen[uid]["sl"] += qty
                continue
            row = {"ma": master["ma"], "ten": master["ten"], "dvt": master["dvt"],
                   "sl": qty, "trangThai": status, "maOcr": code}
        else:
            uid = "?" + key
            if uid in seen:
                seen[uid]["sl"] += qty
                continue
            row = {"ma": code, "ten": "", "dvt": "", "sl": qty,
                   "trangThai": "Không có trong DB", "maOcr": code}
        seen[uid] = row
        items.append(row)
    return items


# ======================================================================
# API
# ======================================================================
@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI PDF & Database Backend is running successfully!"}


# Dùng `def` (không phải `async def`): các tác vụ DB / OpenCV / OCR đều chặn luồng,
# FastAPI sẽ tự chạy chúng trong thread pool thay vì làm nghẽn event loop.
@app.post("/api/process-pdf")
def process_pdf(file: UploadFile = File(...)):
    contents = file.file.read()
    if len(contents) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File vượt quá {MAX_PDF_MB}MB")

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="File không phải PDF hợp lệ")

    db = SessionLocal()
    try:
        master_items = load_master_items(db)
        batch_id = uuid.uuid4().hex
        ocr_database = {}
        customer_cache = {}

        for page_index in range(len(doc)):
            page_no = page_index + 1
            page = doc[page_index]
            warnings = []

            img = render_page(page, PDF_DPI)
            if img is None:
                ocr_database[page_no] = {
                    "ten": "Không đọc được trang", "tienCon": "0 đ",
                    "items": [],
                    "canhBao": ["Không render được trang PDF"],
                }
                continue

            # --- QR code (chính là cột "STT" của bảng _A2) ---
            qr_data = read_qr(img)
            if not qr_data and QR_FALLBACK_DPI > PDF_DPI:
                hi_img = render_page(page, QR_FALLBACK_DPI)
                if hi_img is not None:
                    qr_data = read_qr(hi_img)

            if qr_data:
                if qr_data not in customer_cache:
                    customer_cache[qr_data] = lookup_customer(db, qr_data)
                ten_khach = customer_cache[qr_data]
            else:
                ten_khach = "Không tìm thấy mã QR trên trang"
                warnings.append("Không đọc được QR")

            # --- OCR Mã hàng + Số lượng ---
            ocr_rows, ocr_warnings = extract_table_rows(img)
            warnings.extend(ocr_warnings)
            if not master_items:
                warnings.append("Bảng danh mục hàng hóa đang trống")

            ocr_database[page_no] = {
                "qr": qr_data,
                "ten": ten_khach,
                "tienCon": "0 đ",
                "items": merge_ocr_into_items(master_items, ocr_rows),
                "canhBao": warnings,
            }

        return {"batch_id": batch_id, "total_pages": len(doc), "data": ocr_database}

    except HTTPException:
        raise
    except pytesseract.TesseractNotFoundError:
        log.error("Chưa cài Tesseract trên máy chủ")
        raise HTTPException(status_code=500, detail="Máy chủ chưa cài Tesseract OCR")
    except Exception:
        db.rollback()
        log.error("LỖI XỬ LÝ CHUNG:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Lỗi xử lý file PDF")
    finally:
        doc.close()
        db.close()


class ItemModel(BaseModel):
    ma: str
    ten: str = ""
    dvt: str = ""
    sl: int = Field(0, ge=0)


class PageDataModel(BaseModel):
    batch_id: str = Field(..., min_length=1, max_length=64)
    page: int = Field(..., ge=1)
    ten: str
    tienCon: str = "0 đ"
    items: List[ItemModel]


@app.post("/api/save-page")
def save_page(data: PageDataModel):
    db = SessionLocal()
    try:
        db.execute(
            text("DELETE FROM phieu_mua_hang WHERE batch_id = :batch_id AND page = :page"),
            {"batch_id": data.batch_id, "page": data.page},
        )
        if data.items:
            db.execute(
                text("""
                    INSERT INTO phieu_mua_hang
                        (batch_id, page, ten_pham_nhan, tien_con, ma_hang, ten_hang, dvt, so_luong, trang_thai)
                    VALUES
                        (:batch_id, :page, :ten, :tien_con, :ma_hang, :ten_hang, :dvt, :so_luong, :trang_thai)
                """),
                [
                    {
                        "batch_id": data.batch_id,
                        "page": data.page,
                        "ten": data.ten,
                        "tien_con": data.tienCon,
                        "ma_hang": it.ma,
                        "ten_hang": it.ten,
                        "dvt": it.dvt,
                        "so_luong": it.sl,
                        "trang_thai": "Đã duyệt",
                    }
                    for it in data.items
                ],
            )
        db.commit()
        return {"status": "success", "message": f"Đã lưu thành công dữ liệu trang {data.page} vào cơ sở dữ liệu!"}
    except Exception:
        db.rollback()
        log.error("LỖI KHI LƯU TRANG:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Không lưu được dữ liệu trang")
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
