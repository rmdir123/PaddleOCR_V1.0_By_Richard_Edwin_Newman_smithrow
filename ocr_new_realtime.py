import os, re, json, time, math
import cv2
import numpy as np
from paddleocr import PaddleOCR
import csv

# -------------------- CONFIG --------------------
SHOW_TEXT_ON_BOX = False     # แสดงข้อความบนกรอบไหม (False = ไม่แสดง)
OCR_INTERVAL = 2           # เวลาห่างระหว่างการรัน OCR (วินาที)
ROI_MODE = "smart"           # "smart" = ใช้ ROI รอบก่อน, "center" = ครอปกลางภาพตลอด

# -------------------- PATH --------------------
BASE = r"C:\Users\HP\Desktop\Year4PJ\PaddleOCR_project\PaddleOCR"
OUT_DIR = os.path.join(BASE, "inference_results")
POSTCODE_PATH = r"C:\Users\HP\Desktop\Year4PJ\PaddleOCR_project\PaddleOCR\districts.json"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- POSTAL & CONTEXT --------------------
POSTAL_KEYWORDS = {"รหัสไปรษณีย์","ไปรษณีย์","postcode","postal","zip","zip code","zipcode","post code"}
PHONE_WORDS = {"โทร","tel","phone"}

def load_postcode_whitelist(path: str):
    """
    รองรับทั้ง .json และ .csv
    return: set ของรหัสไปรษณีย์ (str 5 หลัก)
    """
    whitelist = set()
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ---- ปรับตามโครงไฟล์ ----
        if isinstance(data, list):
            # เคส 1: [{"postalCode": "10120", ...}, ...]
            for row in data:
                for key in ("postalCode", "postcode", "zip", "zipcode"):
                    code = str(row.get(key, "")).strip()
                    if code.isdigit() and len(code) == 5:
                        whitelist.add(code)
        else:
            # เคส 2: nested
            def walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in ("postalCode","postcode","zip","zipcode","zip_code"):
                            code = str(v).strip()
                            if code.isdigit() and len(code) == 5:
                                whitelist.add(code)
                        walk(v)
                elif isinstance(obj, list):
                    for it in obj:
                        walk(it)
            walk(data)
    else:  # CSV
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            def pick(row):
                for key in ("postalcode","postcode","zip","zipcode","zip_code","post_code"):
                    if key in row:
                        return row[key]
                for c in row:
                    lc = c.lower()
                    if "zip" in lc or "post" in lc:
                        return row[c]
                return ""
            for r in reader:
                code = str(pick({k.lower(): v for k, v in r.items()})).strip()
                if code.isdigit() and len(code) == 5:
                    whitelist.add(code)
    return whitelist

TH_POST_WHITELIST = load_postcode_whitelist(POSTCODE_PATH)

def is_thai_postal(code: str) -> bool:
    return code in TH_POST_WHITELIST

def find_postal_candidates(text: str):
    digits = re.sub(r"\D", "", text or "")
    return [digits[i:i+5] for i in range(max(0, len(digits)-4)) if len(digits[i:i+5]) == 5]

def box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs)/4.0, sum(ys)/4.0)

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------------- OCR (tuned for realtime) --------------------
ocr = PaddleOCR(
    device="gpu:0",
    use_textline_orientation=True,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="th_PP-OCRv5_mobile_rec",
    text_det_limit_side_len=1280,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=1.6,
    text_rec_score_thresh=0.4
)

# -------------------- Webcam --------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ เปิดกล้องไม่ได้")

# ปรับความละเอียด (ช่วยให้ลื่นขึ้น ถ้ากล้องรองรับ)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_t = 0.0
frame_count = 0
latest_det = None
last_roi = None   # จะเก็บ ROI ล่าสุดไว้ (x1, y1, x2, y2)

print("[INFO] Press 's' to save current frame results, 'q' or ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้ — ข้ามเฟรมนี้")
        time.sleep(0.05)
        continue

    vis = frame.copy()
    H, W = vis.shape[:2]
    diag = math.hypot(W, H)
    NEAR_RADIUS = 0.12 * diag  # สำหรับดูว่าใกล้ keyword / โทรไหม

    now = time.time()
    run_ocr = (now - prev_t) > OCR_INTERVAL

    # -------------------- ทำ OCR เฉพาะบางเฟรม + เฉพาะ ROI --------------------
    if run_ocr:
        prev_t = now

        # 1) เลือก ROI
        if ROI_MODE == "center" or last_roi is None:
            # ครอปกลางภาพ 60% x 55%
            roi_w = int(W * 1)
            roi_h = int(H * 1)
            x1 = (W - roi_w) // 2
            y1 = (H - roi_h) // 2
            x2 = x1 + roi_w
            y2 = y1 + roi_h
        else:
            # ใช้ ROI รอบก่อน
            x1, y1, x2, y2 = last_roi
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W-1, x2); y2 = min(H-1, y2)

        roi = frame[y1:y2, x1:x2].copy()

        t0 = time.time()
        results = ocr.predict(roi)   # ใช้ BGR ROI
        t1 = time.time()

        texts, scores, boxes = [], [], []
        keyword_centers, phone_centers = [], []

        if results and len(results) > 0:
            det = results[0]
            # ---- parse fields ----
            texts  = det.get("rec_texts", []) or det.get("texts", []) or []
            scores = det.get("rec_scores", []) or det.get("scores", []) or []
            raw_boxes = (det.get("boxes", []) or det.get("det_polys", []) or
                         det.get("polys", []) or det.get("dt_polys", []) or
                         det.get("dt_boxes", []) or [])

            # ทำเป็น 4 จุด + แปลงจากพิกัด ROI -> พิกัดภาพใหญ่
            clean_boxes = []
            for b in raw_boxes:
                try:
                    if b is None:
                        continue
                    quad = [(float(p[0]), float(p[1])) for p in b]
                    if len(quad) == 2:  # (x1,y1),(x2,y2)
                        (rx1, ry1), (rx2, ry2) = quad
                        quad = [(rx1, ry1),(rx2, ry1),(rx2, ry2),(rx1, ry2)]
                    if len(quad) == 4:
                        # บวก offset กลับไปตำแหน่งจริง
                        quad = [(px + x1, py + y1) for (px, py) in quad]
                        clean_boxes.append(quad)
                except:
                    pass
            boxes = clean_boxes

            # เก็บ center ของ keyword/phone ในพิกัดภาพใหญ่
            for box, t in zip(boxes, texts):
                t_low = (t or "").lower()
                c = box_center(box)
                if any(kw in t_low for kw in POSTAL_KEYWORDS):
                    keyword_centers.append(c)
                if any(pw in t_low for pw in PHONE_WORDS):
                    phone_centers.append(c)

            # ---- Ranking ผู้สมัครรหัสไปรษณีย์ด้วยบริบท ----
            postal_candidates = []
            for idx, (t, s, box) in enumerate(zip(texts, scores, boxes), start=1):
                conf = float(s) if isinstance(s, (int, float)) else 0.0
                t_low = (t or "").lower()
                cx, cy = box_center(box)

                for code in find_postal_candidates(t):
                    valid_whitelist = (code in TH_POST_WHITELIST)
                    score = 0
                    if valid_whitelist: score += 4
                    if keyword_centers:
                        d = min(l2((cx,cy), kc) for kc in keyword_centers)
                        if d <= NEAR_RADIUS:
                            score += 2
                    if cy > H * 0.55: score += 1
                    if cx > W * 0.55: score += 1
                    score += int(round(conf * 2))
                    if phone_centers:
                        dph = min(l2((cx,cy), pc) for pc in phone_centers)
                        if dph <= NEAR_RADIUS:
                            score -= 2
                    if any(kw in t_low for kw in POSTAL_KEYWORDS):
                        score += 1

                    postal_candidates.append({
                        "line_index": idx,
                        "text": t,
                        "ocr_score": conf,
                        "code": code,
                        "valid_th_range": valid_whitelist,
                        "score": int(score),
                        "center": (cx, cy)
                    })

            best_postal = None
            if postal_candidates:
                postal_candidates.sort(
                    key=lambda c: (c["score"], c["ocr_score"], c["center"][1], c["center"][0]),
                    reverse=True
                )
                best_postal = postal_candidates[0]
            else:
                postal_candidates = []

            # อัปเดต ROI อัตโนมัติจากกล่องที่เจอ
            if boxes:
                xs = [p[0] for box in boxes for p in box]
                ys = [p[1] for box in boxes for p in box]
                rx1, ry1, rx2, ry2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                pad = 15
                rx1 = max(0, rx1 - pad)
                ry1 = max(0, ry1 - pad)
                rx2 = min(W-1, rx2 + pad)
                ry2 = min(H-1, ry2 + pad)
                last_roi = (rx1, ry1, rx2, ry2)
            else:
                # ถ้าไม่เจอเลย กลับไปใช้ center รอบหน้า
                last_roi = None

            latest_det = {
                "texts": texts,
                "scores": scores,
                "boxes": boxes,
                "postal_candidates": postal_candidates,
                "best_postal": best_postal,
                "latency_ms": int((t1 - t0) * 1000)
            }

    # -------------------- วาดผล --------------------
    if latest_det:
        texts  = latest_det["texts"]
        boxes  = latest_det["boxes"]
        best   = latest_det["best_postal"]
        cand_lines = set()
        good_lines = set()
        for c in latest_det["postal_candidates"] or []:
            cand_lines.add(c["line_index"])
            if is_thai_postal(c["code"]):
                good_lines.add(c["line_index"])

        for i, (box, t) in enumerate(zip(boxes, texts), start=1):
            pts = np.array(box, dtype=np.int32)
            color = (0, 200, 0)          # เขียว = ทั่วไป
            if i in cand_lines:
                color = (0, 165, 255)    # ส้ม = เป็น candidate
            if best and i == best["line_index"]:
                color = (0, 0, 255)      # แดง = คาดว่าเป็นรหัสไปรษณีย์
            cv2.polylines(vis, [pts], True, color, 2)

            if SHOW_TEXT_ON_BOX:
                x, y = int(pts[0][0]), int(pts[0][1]) - 5
                cv2.putText(vis, f"[{i}] {t}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

        # ข้อมูลด้านบนซ้าย
        y0 = 30
        cv2.putText(vis, f"OCR latency: {latest_det['latency_ms']} ms", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y0 += 28
        if latest_det["best_postal"]:
            b = latest_det["best_postal"]
            cv2.putText(vis, f"POSTAL: {b['code']} (line {b['line_index']})", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.putText(vis, "POSTAL: -", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

    # วาด ROI ที่กำลังจะ/เพิ่งใช้ OCR ให้เห็น
    if last_roi is not None:
        x1, y1, x2, y2 = last_roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 1)

    cv2.imshow("Thai OCR (Realtime)  -  's': save frame  |  'q'/ESC: quit", vis)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):   # ESC หรือ q
        break
    if key == ord('s'):
        frame_count += 1
        stem = f"cam_{frame_count:04d}"
        img_path = os.path.join(OUT_DIR, f"{stem}.jpg")
        cv2.imwrite(img_path, vis)
        print(f"[SAVE] image -> {img_path}")

        if latest_det:
            txt_path = os.path.join(OUT_DIR, f"{stem}.txt")
            json_path = os.path.join(OUT_DIR, f"{stem}.json")
            with open(txt_path, "w", encoding="utf-8") as f:
                for t, s in zip(latest_det["texts"], latest_det["scores"]):
                    if s is None:
                        f.write(f"{t}\tna\n")
                    else:
                        f.write(f"{t}\t{float(s):.3f}\n")
                f.write("\n[POSTAL]\n")
                if latest_det["best_postal"]:
                    b = latest_det["best_postal"]
                    f.write(f"BEST\t{b['code']}\tline={b['line_index']}\tscore={b['score']}\n")
                else:
                    f.write("NONE\n")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "image": os.path.basename(img_path),
                    "best_postal": latest_det["best_postal"],
                    "postal_candidates": latest_det["postal_candidates"]
                }, jf, ensure_ascii=False, indent=2)
            print(f"[SAVE] txt  -> {txt_path}")
            print(f"[SAVE] json -> {json_path}")

cap.release()
cv2.destroyAllWindows()
