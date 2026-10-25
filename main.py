import copy
import difflib
import logging
import os
import re
import threading
import time
import traceback
import unicodedata
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import List
from zoneinfo import ZoneInfo

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

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

PDF_DPI = int(os.environ.get("PDF_DPI", "200"))
QR_FALLBACK_DPI = int(os.environ.get("QR_FALLBACK_DPI", "300"))
MAX_PDF_MB = int(os.environ.get("MAX_PDF_MB", "30"))
OCR_LANG_WANTED = os.environ.get("OCR_LANG", "vie+eng")
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "20"))

try:
    VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    VN_TZ = timezone(timedelta(hours=7))

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

SCHEMA_STATEMENTS = [
    "ALTER TABLE phieu_mua_hang ADD COLUMN IF NOT EXISTS batch_id TEXT",
    "CREATE INDEX IF NOT EXISTS idx_phieu_batch_page ON phieu_mua_hang (batch_id, page)",
    """
    CREATE TABLE IF NOT EXISTS phieu_chi (
        id BIGSERIAL PRIMARY KEY,
        batch_id TEXT NOT NULL,
        page INTEGER NOT NULL,
        loai TEXT NOT NULL DEFAULT 'Chi',
        stt TEXT NOT NULL,
        ngay DATE NOT NULL,
        ho_ten TEXT,
        ngay_sinh TEXT,
        gioi_tinh TEXT,
        noi_tt TEXT,
        hanh_vi TEXT,
        ngay_bat TEXT,
        buong_giam TEXT,
        tgiam TEXT,
        tong_so_luong INTEGER NOT NULL DEFAULT 0,
        so_tien BIGINT NOT NULL DEFAULT 0,
        so_tien_bang_chu TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (batch_id, page)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS phieu_chi_ct (
        id BIGSERIAL PRIMARY KEY,
        phieu_id BIGINT NOT NULL REFERENCES phieu_chi(id) ON DELETE CASCADE,
        ma_hang TEXT NOT NULL,
        ten_hang TEXT,
        dvt TEXT,
        so_luong INTEGER NOT NULL,
        gia BIGINT NOT NULL,
        thanh_tien BIGINT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_phieu_chi_ct_phieu ON phieu_chi_ct (phieu_id)",
    "CREATE INDEX IF NOT EXISTS idx_phieu_chi_stt_ngay ON phieu_chi (stt, ngay)",
]


@app.on_event("startup")
def ensure_schema():
    for stmt in SCHEMA_STATEMENTS:
        db = SessionLocal()
        try:
            db.execute(text(stmt))
            db.commit()
        except Exception:
            db.rollback()
            log.warning("Lệnh schema lỗi (bỏ qua): %s\n%s", " ".join(stmt.split())[:70], traceback.format_exc())
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
# OCR BẢNG
# ======================================================================
def remove_table_lines(gray: np.ndarray) -> np.ndarray:
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = ink.shape[:2]
    horiz = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 40), 1))
    )
    vert = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 40, 40)))
    )
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8), iterations=1)
    cleaned = ink.copy()
    cleaned[lines > 0] = 0
    return 255 - cleaned


_HDR_FIX = str.maketrans({"0": "o", "6": "o", "1": "l", "5": "s", "8": "b", "3": "e", "&": "a", "@": "a"})


def header_norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", strip_accents(s).translate(_HDR_FIX))


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
            "text": t, "n": header_norm(t),
            "x": x, "y": y, "w": w, "h": h,
            "cx": x + w / 2, "cy": y + h / 2,
        })
    return words


def find_header_pair(words: list, first: str, second: str):
    best = None
    merged = first + second
    for a in words:
        if len(a["n"]) >= len(merged) - 1 and sim(a["n"], merged) >= 0.8:
            if best is None or a["cy"] < best[0]["cy"]:
                best = (a, a)
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
    inset = 6
    x0, x1 = max(x0 + inset, 0), max(x1 - inset, 0)
    y0, y1 = max(y0, 0), max(y1, 0)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(bw, lang="eng", config=cfg)
    return re.sub(r"\D", "", txt)


def extract_table_rows(img: np.ndarray):
    warnings = []
    img = remove_table_lines(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
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
# TRUY VẤN DATABASE & TIỆN ÍCH PHỤ
# ======================================================================
MASTER_TABLE = "danh muc hang trai 2"


def _find_column(columns: list, *candidates: str):
    def key(s: str) -> str:
        return strip_accents(s).replace(" ", "").replace("_", "")

    wanted = {key(c) for c in candidates}
    for col in columns:
        if key(col) in wanted:
            return col
    return None


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def parse_money(v) -> int:
    if v is None:
        return 0
    if isinstance(v, float) and v != v:
        return 0
    if isinstance(v, (int, float, Decimal)):
        return int(round(v))
    digits = re.sub(r"[^0-9]", "", str(v))
    return int(digits) if digits else 0


def fmt_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, date):
        return v.strftime("%d/%m/%Y")
    s = str(v).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    return s


def get_columns(db, table: str) -> list:
    rows = db.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :t ORDER BY ordinal_position"
        ),
        {"t": table},
    ).fetchall()
    return [r[0] for r in rows]


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read_triple(n: int, full: bool) -> str:
    tr, ch, dv = n // 100, (n % 100) // 10, n % 10
    parts = []
    if tr > 0 or full:
        parts.append(_DIGITS[tr] + " trăm")
    if ch > 1:
        parts.append(_DIGITS[ch] + " mươi")
    elif ch == 1:
        parts.append("mười")
    elif dv > 0 and (tr > 0 or full):
        parts.append("lẻ")
    if dv > 0:
        if dv == 1 and ch > 1:
            parts.append("mốt")
        elif dv == 5 and ch > 0:
            parts.append("lăm")
        else:
            parts.append(_DIGITS[dv])
    return " ".join(parts)


def money_to_words(amount: int) -> str:
    if amount <= 0:
        return "Không đồng"
    if amount >= 10**12:
        return f"{amount} đồng"
    units = ["", "nghìn", "triệu", "tỷ"]
    chunks, n = [], amount
    while n > 0:
        chunks.append(n % 1000)
        n //= 1000
    top = len(chunks) - 1
    out = []
    for i in range(top, -1, -1):
        if chunks[i] == 0:
            continue
        out.append(_read_triple(chunks[i], full=(i != top)))
        if units[i]:
            out.append(units[i])
    s = " ".join(out)
    return s[:1].upper() + s[1:] + " đồng"


def load_master_items(db) -> list:
    try:
        columns = get_columns(db, MASTER_TABLE)
        col_ma = _find_column(columns, "Mã hàng", "Mã số hàng", "Mã")
        col_ten = _find_column(columns, "Tên hàng", "Tên hàng hóa", "Tên")
        col_dvt = _find_column(columns, "Đơn vị tính", "Đơn vị", "ĐVT", "DVT")
        col_gia = _find_column(columns, "Giá bán", "Đơn giá", "Giá")
        col_status = _find_column(columns, "status", "Trạng thái")

        if not columns or not col_ma or not col_ten:
            raise HTTPException(
                status_code=500,
                detail="Bảng danh mục không có cột Mã hàng hoặc Tên hàng",
            )

        dvt_sql = _quote(col_dvt) if col_dvt else "''"
        gia_sql = _quote(col_gia) if col_gia else "NULL"
        status_sql = _quote(col_status) if col_status else "'on'"
        
        rows = db.execute(
            text(
                f"SELECT {_quote(col_ma)}, {_quote(col_ten)}, {dvt_sql}, {gia_sql}, {status_sql} "
                f"FROM {_quote(MASTER_TABLE)}"
            )
        ).fetchall()
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        log.error("Lỗi đọc bảng danh mục:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Không đọc được bảng danh mục hàng hóa")
    return [
        {
            "ma": str(r[0] or "").strip(),
            "ten": str(r[1] or "").strip(),
            "dvt": str(r[2] or "").strip(),
            "gia": parse_money(r[3]),
            "on": str(r[4] or "").strip().lower() == "on",
            "sl": 0,
            "trangThai": "Chưa đọc được",
        }
        for r in rows
    ]


PERSON_FIELDS = {
    "hoTen": ("Họ và tên", "Họ tên"),
    "ngaySinh": ("Ngày sinh",),
    "gioiTinh": ("Giới tính",),
    "noiTT": ("Nơi TT", "Nơi thường trú"),
    "hanhVi": ("Hành vi",),
    "ngayBat": ("Ngày bắt",),
    "buongGiam": ("Buồng giam", "Buồng"),
    "tGiam": ("TGiam", "Tạm giam"),
}
_PERSON_COLS = {"at": 0.0, "value": None}
_PERSON_COLS_LOCK = threading.Lock()


def person_columns(db):
    with _PERSON_COLS_LOCK:
        if _PERSON_COLS["value"] and time.time() - _PERSON_COLS["at"] < 300:
            return _PERSON_COLS["value"]
    columns = get_columns(db, "_A2")
    col_stt = _find_column(columns, "STT")
    picked = {}
    for key, names in PERSON_FIELDS.items():
        col = _find_column(columns, *names)
        if col:
            picked[key] = col
    if not col_stt or "hoTen" not in picked:
        raise HTTPException(status_code=500, detail="Bảng _A2 thiếu cột STT hoặc Họ và tên")
    
    value = (col_stt, picked)
    with _PERSON_COLS_LOCK:
        _PERSON_COLS["at"] = time.time()
        _PERSON_COLS["value"] = value
    return value


def fetch_person(db, stt: str):
    col_stt, picked = person_columns(db)
    select_cols = ", ".join(_quote(c) for c in picked.values())
    row = db.execute(
        text(f'SELECT {select_cols} FROM "_A2" WHERE {_quote(col_stt)}::text = :stt LIMIT 1'),
        {"stt": stt},
    ).fetchone()
    if row is None:
        return None
    person = {k: "" for k in PERSON_FIELDS}
    for key, value in zip(picked.keys(), row):
        person[key] = fmt_value(value)
    return person


_CONFUSION = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"})


def _confuse(key: str) -> str:
    return key.translate(_CONFUSION)


def merge_ocr_into_items(master_items: list, ocr_rows: list) -> list:
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
                   "gia": master["gia"], "sl": qty, "trangThai": status, "maOcr": code}
        else:
            uid = "?" + key
            if uid in seen:
                seen[uid]["sl"] += qty
                continue
            row = {"ma": code, "ten": "", "dvt": "", "gia": 0, "sl": qty,
                   "trangThai": "Không có trong DB", "maOcr": code}
        seen[uid] = row
        items.append(row)
    return items


# ======================================================================
# API ENDPOINTS
# ======================================================================
@app.get("/")
def read_root():
    return {"status": "online", "message": "FastAPI PDF & Database Backend is running successfully!"}


def analyze_page(page, master_items: list, db, person_cache: dict) -> dict:
    warnings = []
    img = render_page(page, PDF_DPI)
    if img is None:
        return {
            "qr": None, "ten": "Không đọc được trang", "nguoi": None, "tienCon": "0 đ",
            "items": [], "canhBao": ["Không render được trang PDF"],
        }

    qr_data = read_qr(img)
    if not qr_data and QR_FALLBACK_DPI > PDF_DPI:
        hi_img = render_page(page, QR_FALLBACK_DPI)
        if hi_img is not None:
            qr_data = read_qr(hi_img)

    nguoi = None
    if qr_data:
        if qr_data not in person_cache:
            try:
                person_cache[qr_data] = fetch_person(db, qr_data)
            except HTTPException:
                raise
            except Exception:
                db.rollback()
                log.error("Lỗi truy vấn _A2 (STT=%s):\n%s", qr_data, traceback.format_exc())
                person_cache[qr_data] = False
        cached = person_cache[qr_data]
        if cached is False:
            ten_khach = "Lỗi truy vấn bảng _A2"
            warnings.append("Lỗi truy vấn bảng _A2")
        elif cached is None:
            ten_khach = f"Mã QR [{qr_data}] không tồn tại trong bảng _A2"
            warnings.append(f"STT {qr_data} không có trong bảng _A2")
        else:
            nguoi = cached
            ten_khach = cached["hoTen"]
    else:
        ten_khach = "Không tìm thấy mã QR trên trang"
        warnings.append("Không đọc được QR")

    ocr_rows, ocr_warnings = extract_table_rows(img)
    warnings.extend(ocr_warnings)
    if not master_items:
        warnings.append("Bảng danh mục hàng hóa đang trống")

    return {
        "qr": qr_data,
        "ten": ten_khach,
        "nguoi": nguoi,
        "tienCon": "0 đ",
        "items": merge_ocr_into_items(master_items, ocr_rows),
        "canhBao": warnings,
    }


def analyze_pdf(contents: bytes, master_items: list, db, on_total=None, on_page=None) -> dict:
    doc = fitz.open(stream=contents, filetype="pdf")
    try:
        total = len(doc)
        if on_total:
            on_total(total)
        pages, person_cache = {}, {}
        for i in range(total):
            result = analyze_page(doc[i], master_items, db, person_cache)
            pages[i + 1] = result
            if on_page:
                on_page(i + 1, result)
        return pages
    finally:
        doc.close()


def read_pdf_upload(file: UploadFile):
    contents = file.file.read()
    if len(contents) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File vượt quá {MAX_PDF_MB}MB")
    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        total = len(doc)
        doc.close()
    except Exception:
        raise HTTPException(status_code=400, detail="File không phải PDF hợp lệ")
    return contents, total


OCR_SEMAPHORE = threading.Semaphore(1)


@app.post("/api/process-pdf")
def process_pdf(file: UploadFile = File(...)):
    contents, _total = read_pdf_upload(file)
    db = SessionLocal()
    try:
        master_items = load_master_items(db)
        with OCR_SEMAPHORE:
            pages = analyze_pdf(contents, master_items, db)
        return {"batch_id": uuid.uuid4().hex, "total_pages": len(pages), "data": pages}
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
        db.close()


BATCHES: dict = {}
BATCHES_LOCK = threading.Lock()


def _run_batch(batch_id: str, contents: bytes):
    job = BATCHES.get(batch_id)
    if job is None:
        return
    db = SessionLocal()
    try:
        with OCR_SEMAPHORE:
            master_items = load_master_items(db)

            def on_total(n):
                job["total"] = n

            def on_page(n, result):
                with BATCHES_LOCK:
                    job["pages"][n] = result
                    job["done"] = len(job["pages"])

            analyze_pdf(contents, master_items, db, on_total, on_page)
        job["status"] = "done"
    except HTTPException as e:
        job["status"], job["error"] = "error", str(e.detail)
    except pytesseract.TesseractNotFoundError:
        log.error("Chưa cài Tesseract trên máy chủ")
        job["status"], job["error"] = "error", "Máy chủ chưa cài Tesseract OCR"
    except Exception:
        db.rollback()
        log.error("LỖI XỬ LÝ LÔ %s:\n%s", batch_id, traceback.format_exc())
        job["status"], job["error"] = "error", "Lỗi xử lý file PDF"
    finally:
        db.close()


@app.post("/api/batches")
def create_batch(file: UploadFile = File(...)):
    contents, total = read_pdf_upload(file)
    batch_id = uuid.uuid4().hex
    job = {"status": "running", "total": total, "done": 0, "error": None,
           "pages": {}, "created": time.time()}
    with BATCHES_LOCK:
        BATCHES[batch_id] = job
        while len(BATCHES) > MAX_BATCHES:
            oldest = min(BATCHES, key=lambda k: BATCHES[k]["created"])
            BATCHES.pop(oldest, None)
    threading.Thread(target=_run_batch, args=(batch_id, contents), daemon=True).start()
    return {"batch_id": batch_id, "total_pages": total}


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str, since: int = 0):
    with BATCHES_LOCK:
        job = BATCHES.get(batch_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy lô PDF (có thể máy chủ đã khởi động lại). Hãy mở lại file PDF.",
            )
        pages = {n: r for n, r in job["pages"].items() if n > since}
        return {
            "status": job["status"], "total": job["total"], "done": job["done"],
            "error": job["error"], "pages": pages,
        }


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


@app.get("/api/products")
def get_products():
    db = SessionLocal()
    try:
        items = load_master_items(db)
    finally:
        db.close()
    data = [
        {"ma": i["ma"], "ten": i["ten"], "dvt": i["dvt"], "gia": i["gia"]}
        for i in items
        if i["on"] and i["ma"]
    ]
    data.sort(key=lambda x: natural_key(x["ma"]))
    return {"count": len(data), "data": data}


@app.get("/api/person/{stt}")
def get_person(stt: str):
    db = SessionLocal()
    try:
        person = fetch_person(db, stt.strip())
        if not person:
            raise HTTPException(status_code=404, detail="Không tìm thấy thông tin can phạm với STT này")
        return {"status": "success", "data": person}
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        log.error("Lỗi truy vấn _A2 (STT=%s):\n%s", stt, traceback.format_exc())
        raise HTTPException(status_code=500, detail="Lỗi truy vấn cơ sở dữ liệu")
    finally:
        db.close()
