import os, glob, math, textwrap
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

BASE = r"C:\Users\HP\Desktop\Year4PJ\PaddleOCR_project\PaddleOCR"
IMG_DIR = os.path.join(BASE, "test_images")
OUT_DIR = os.path.join(BASE, "inference_results")
os.makedirs(OUT_DIR, exist_ok=True)

#get Thai fonts
def pick_thai_font(size=22):
    candidates = [
        r"C:\Windows\Fonts\THSarabunNew.ttf",
        r"C:\Windows\Fonts\LeelawUI.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\Angsa.ttf",
        r"C:\Windows\Fonts\Cordia.ttf",
    ]
    for f in candidates:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size=size)
            except:
                pass
    return ImageFont.load_default()

FONT_MAIN = pick_thai_font(24)
FONT_SMALL = pick_thai_font(18)
FONT_TINY = pick_thai_font(16)

#get all images type
paths = []
for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
    paths += glob.glob(os.path.join(IMG_DIR, ext))
print(f"[SMOKE] found {len(paths)} images")
if not paths:
    raise SystemExit("no images in test_images")

#create OCR
ocr = PaddleOCR(
    device="gpu:0",  
    use_textline_orientation=False,       # close CLS
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="th_PP-OCRv5_mobile_rec"
)


def draw_number_badge(draw, center, text, fill=(0, 122, 255), text_color=(255,255,255)):
    r = 14
    x, y = center
    draw.ellipse((x-r, y-r, x+r, y+r), fill=fill)
    tw, th = draw.textbbox((0,0), text, font=FONT_TINY)[2:]
    draw.text((x - tw/2, y - th/2), text, font=FONT_TINY, fill=text_color)

def wrap_text_by_width(draw, text, font, max_width):

    if not text: return [""]
    lines, line = [], ""
    for ch in text:
        test = line + ch
        w = draw.textbbox((0,0), test, font=font)[2]
        if w <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
                line = ch
            else:
                lines.append(ch)
                line = ""
    if line:
        lines.append(line)
    return lines

for img in paths:
    print(f"\n[OCR] run: {img}")
    result = ocr.predict(img)

    stem = os.path.splitext(os.path.basename(img))[0]
    out_img_path = os.path.join(OUT_DIR, f"{stem}_ocr.jpg")
    out_txt_path = os.path.join(OUT_DIR, f"{stem}.txt")

    pil_im = Image.open(img).convert("RGB")
    W, H = pil_im.size

    #create space for result in jpg
    sidebar_w = max(420, int(W * 0.34))
    canvas = Image.new("RGB", (W + sidebar_w, H), (255,255,255))
    canvas.paste(pil_im, (0,0))
    draw = ImageDraw.Draw(canvas)

 
    draw.line([(W,0), (W,H)], fill=(230,230,230), width=2)

    #get text result
    texts, scores, boxes = [], [], []
    if result and len(result) > 0:
        det = result[0]
        texts = det.get("rec_texts", []) or []
        scores = det.get("rec_scores", []) or []
        boxes  = det.get("boxes", []) or []

    #create .txt for output
    with open(out_txt_path, "w", encoding="utf-8") as f:
        if texts:
            for t, s in zip(texts, scores):
                f.write(f"{t}\t{s:.3f}\n")
        else:
            f.write("[NO_TEXT]\n")

    #create detect frame
    for i, box in enumerate(boxes):
        try:
            pts = [(int(p[0]), int(p[1])) for p in box]
            #creat detected polygon
            draw.line([tuple(pts[0]), tuple(pts[1]), tuple(pts[2]), tuple(pts[3]), tuple(pts[0])],
                      width=3, fill=(0, 200, 0))
    
            cx = int((pts[0][0] + pts[1][0]) / 2)
            cy = int((pts[0][1] + pts[1][1]) / 2) - 18
            cx = max(18, min(cx, W-18))
            cy = max(18, min(cy, H-18))
            draw_number_badge(draw, (cx, cy), str(i+1))
        except:
            pass

    #result info gen
    panel_x = W + 20
    y = 16
    title = f"Thai OCR — {os.path.basename(img)}"
    draw.text((panel_x, y), title, font=FONT_MAIN, fill=(0,0,0))
    y += draw.textbbox((0,0), title, font=FONT_MAIN)[3] + 10

    if not texts:
        draw.text((panel_x, y), "ไม่พบข้อความ", font=FONT_MAIN, fill=(150,0,0))
    else:
        max_text_width = sidebar_w - 40
        for i, (t, s) in enumerate(zip(texts, scores), start=1):
            #score
            head = f"[{i}]  score {s:.3f}"
            draw.text((panel_x, y), head, font=FONT_SMALL, fill=(60,60,60))
            y += draw.textbbox((0,0), head, font=FONT_SMALL)[3] + 2

            #detected text
            lines = wrap_text_by_width(draw, t, FONT_SMALL, max_text_width)
            for ln in lines:
                draw.text((panel_x, y), ln, font=FONT_SMALL, fill=(0,0,0))
                y += draw.textbbox((0,0), ln, font=FONT_SMALL)[3] + 2
            y += 6  # space

            #got x line more 
            if y > H - 40:
                more = f"... ({len(texts) - i} lines more)"
                draw.text((panel_x, H - 28), more, font=FONT_SMALL, fill=(120,0,0))
                break

    canvas.save(out_img_path, quality=95)
    print(f"[OCR] saved image: {out_img_path}")
    print(f"[OCR] saved text : {out_txt_path}")

print("\n[DONE] All results saved to:", OUT_DIR)
