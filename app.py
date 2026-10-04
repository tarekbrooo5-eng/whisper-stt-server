import os
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper

app = Flask(__name__)
CORS(app)

# تحميل نموذج Whisper (يُفضل استخدام نموذج "tiny" أو "base" لتناسب قيود الذاكرة المجانية على Render)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded successfully!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "message": "Video Transcription Server is Live!"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    input_video_path = None
    output_audio_path = None
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # حفظ الفيديو المرفوع مؤقتاً
        input_video_path = "temp_input_video.mp4"
        output_audio_path = "temp_extracted_audio.wav"
        file.save(input_video_path)

        # 1. استخراج الصوت باستخدام FFmpeg بصيغة WAV بتردد 16kHz (المعيار المثالي لـ Whisper)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_video_path, 
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_audio_path
        ]
        
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # 2. تشغيل نموذج Whisper لاستخراج النصوص والطوابع الزمنية (Word-Level / Segment-Level Timestamps)
        result = model.transcribe(output_audio_path, task="transcribe", word_timestamps=True)

        # تجهيز المقاطع والكلمات للإرسال إلى التطبيق
        segments_data = []
        for segment in result.get("segments", []):
            segments_data.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip()
            })

        return jsonify({
            "success": True,
            "segments": segments_data
        })

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg error: {e.stderr.decode('utf-8')}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # تنظيف الملفات المؤقتة من السيرفر للحفاظ على الذاكرة
        if input_video_path and os.path.exists(input_video_path):
            os.remove(input_video_path)
        if output_audio_path and os.path.exists(output_audio_path):
            os.remove(output_audio_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
