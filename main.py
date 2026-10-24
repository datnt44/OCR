# ĐỌC MÃ QR CODE (CHÍNH LÀ "STT" TRONG BẢNG _A2 ĐỂ LẤY "Họ và tên")
            qr_codes = decode_qr(img)
            ten_khach = "Không tìm thấy mã QR trên trang"
            tien_con = "0 đ"
            
            if qr_codes:
                qr_data = qr_codes[0].data.decode('utf-8').strip()
                try:
                    # Truy vấn chính xác: Lọc theo cột "STT" và lấy cột "Họ và tên" (và cột tiền nếu có)
                    # Lưu ý: Nếu tên cột trong DB của bạn viết khác (ví dụ "ho_ten" hoặc "Họ và tên"), hãy điều chỉnh trong ngoặc kép
                    sql_a2 = text('SELECT "Họ và tên", "Số tiền" FROM "_A2" WHERE "STT"::text = :stt LIMIT 1;')
                    a2_result = db.execute(sql_a2, {"stt": qr_data}).fetchone()
                    
                    if a2_result:
                        # Lấy trực tiếp giá trị từ cột "Họ và tên" (index 0) và cột tiền (index 1)
                        ho_ten = a2_result[0]
                        so_tien = a2_result[1] if len(a2_result) > 1 else 0
                        
                        ten_khach = str(ho_ten or f"STT: {qr_data} (Tên trống)")
                        tien_con = str(so_tien or "0") + " đ"
                    else:
                        ten_khach = f"Mã QR [{qr_data}] không tồn tại trong bảng _A2"
                except Exception as db_err:
                    db.rollback() 
                    # Fallback trường hợp tên cột trong DB viết không dấu hoặc khác một chút
                    try:
                        sql_fallback = text('SELECT * FROM "_A2" WHERE "STT"::text = :stt LIMIT 1;')
                        row_fb = db.execute(sql_fallback, {"stt": qr_data}).fetchone()
                        if row_fb:
                            row_dict = row_fb._asdict()
                            # Tìm tường minh cột có chữ "tên" hoặc "ten"
                            name_val = None
                            money_val = None
                            for k, v in row_dict.items():
                                if "tên" in k.lower() or "ten" in k.lower():
                                    name_val = v
                                if "tiền" in k.lower() or "tien" in k.lower() or "con" in k.lower():
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
