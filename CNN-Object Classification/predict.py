import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont

IMG_SIZE = (128, 128)

def get_font(size=40):
    """โหลด Font สำหรับเขียนข้อความ"""
    try:
        # ลองใช้ Font ที่มีในระบบ
        font = ImageFont.truetype("arial.ttf", size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except:
            font = ImageFont.load_default()
    return font

def predict_and_annotate(model, class_names, image_path, output_dir):
    """ทำนายและวาดกรอบ + ข้อความลงบนรูป"""
    
    # โหลดรูปต้นฉบับ (ไม่ Resize)
    original_img = Image.open(image_path).convert('RGB')
    width, height = original_img.size
    
    # Resize สำหรับ Predict
    img_resized = original_img.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, 0)

    # ทำนาย
    predictions = model.predict(img_array, verbose=0)
    score = predictions[0]
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    # สร้าง Copy ของภาพต้นฉบับเพื่อวาดทับ
    annotated_img = original_img.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # กำหนดสีตามผลทาย
    if predicted_class == "bottle":
        box_color = (0, 255, 0)  # เขียว
    else:
        box_color = (255, 165, 0)  # ส้ม
    
    # วาดกรอบรอบวัตถุ (ประมาณ 80% ของภาพตรงกลาง)
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    box = [margin_x, margin_y, width - margin_x, height - margin_y]
    draw.rectangle(box, outline=box_color, width=5)
    
    # เขียนข้อความผลทาย
    font = get_font(50)
    label = f"{predicted_class.upper()} ({confidence:.1f}%)"
    
    # พื้นหลังข้อความ
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = margin_x
    text_y = margin_y - text_height - 10
    if text_y < 0:
        text_y = margin_y + 10
    
    # วาดพื้นหลังข้อความ
    draw.rectangle(
        [text_x - 5, text_y - 5, text_x + text_width + 10, text_y + text_height + 5],
        fill=box_color
    )
    
    # วาดข้อความ
    draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
    
    # บันทึกรูป
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_result{ext}")
    annotated_img.save(output_path)
    
    return predicted_class, confidence, output_path

def predict_folder_visual(folder_path):
    """ทำนายรูปทั้งโฟลเดอร์และบันทึกเป็นภาพ Output"""
    
    # โหลดโมเดล
    model_path = 'object_classifier.keras'
    if not os.path.exists(model_path):
        print(f"ไม่พบไฟล์โมเดล '{model_path}' กรุณารัน train.py ก่อน")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # โหลดชื่อคลาส
    with open('class_names.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    # สร้างโฟลเดอร์ Output
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # หารูปภาพในโฟลเดอร์
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"ไม่พบรูปภาพในโฟลเดอร์ '{folder_path}'")
        return

    print("\n" + "="*60)
    print(f"กำลังประมวลผล {len(images)} รูปภาพ...")
    print("="*60)
    
    results = []
    for i, img_name in enumerate(sorted(images), 1):
        img_path = os.path.join(folder_path, img_name)
        predicted_class, confidence, output_path = predict_and_annotate(
            model, class_names, img_path, output_dir
        )
        results.append({
            'file': img_name,
            'prediction': predicted_class,
            'confidence': confidence,
            'output': output_path
        })
        print(f"[{i}/{len(images)}] {img_name} → {predicted_class} ({confidence:.1f}%) ✓")
    
    print("\n" + "="*60)
    print("📊 สรุปผลการทดสอบ:")
    for class_name in class_names:
        count = sum(1 for r in results if r['prediction'] == class_name)
        print(f"   - {class_name}: {count} รูป")
    
    print(f"\n✅ บันทึกรูปผลลัพธ์ทั้งหมดในโฟลเดอร์: {output_dir}/")
    print("="*60)

def predict_single_visual(image_path):
    """ทำนายรูปเดียวและบันทึกผลเป็นภาพ"""
    
    if not os.path.exists(image_path):
        print(f"ไม่พบไฟล์ '{image_path}'")
        return

    model_path = 'object_classifier.keras'
    if not os.path.exists(model_path):
        print(f"ไม่พบไฟล์โมเดล '{model_path}' กรุณารัน train.py ก่อน")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    with open('class_names.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    predicted_class, confidence, output_path = predict_and_annotate(
        model, class_names, image_path, output_dir
    )
    
    print(f"\n✅ ผลการทำนาย: {predicted_class} ({confidence:.1f}%)")
    print(f"📁 บันทึกรูปผลลัพธ์: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("วิธีใช้:")
        print("  ทดสอบรูปเดียว:    python predict.py <path_to_image>")
        print("  ทดสอบทั้งโฟลเดอร์: python predict.py <path_to_folder>")
        print("\nตัวอย่าง:")
        print("  python predict.py test_bottle.jpg")
        print("  python predict.py test/")
        print("\nผลลัพธ์จะถูกบันทึกในโฟลเดอร์ 'output/'")
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            predict_folder_visual(path)
        else:
            predict_single_visual(path)
