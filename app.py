import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile

app = Flask(__name__)
CORS(app)  # السماح بالاتصال من أي واجهة أمامية (GitHub Pages)

# تحميل نموذج Whisper (يمكنك استخدام tiny أو base أو small حسب قوة الاستضافة)
# 'base' هو خيار ممتاز وسريع ويدعم اللغة العربية بدقة عالية
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Whisper STT Server is running successfully!"})

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No video or audio file provided"}), 400
    
    file = request.files["file"]
    
    # حفظ الملف مؤقتاً في السيرفر
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        # تشغيل نموذج Whisper لاستخراج النصوص مع التوقيتات (Segments)
        result = model.transcribe(temp_path, task="transcribe")
        
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip()
            })
            
        return jsonify({
            "success": True,
            "text": result.get("text", "").strip(),
            "segments": segments
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        # حذف الملف المؤقت لتوفير مساحة السيرفر
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.5.0" if 'RENDER' in os.environ else "127.0.0.1", port=port)