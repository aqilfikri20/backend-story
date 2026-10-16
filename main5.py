from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.responses import PlainTextResponse, HTMLResponse
import random
import json
import os
import re
import time
import requests

app = FastAPI()

load_dotenv()

api_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

# =========================
# Schema
# =========================
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# =========================
# Struktur cerita
# =========================
bagian_list = [
    "orientasi 1",
    "orientasi 2",
    "konflik awal 1",
    "konflik awal 2",
    "eskalasi 1",
    "eskalasi 2",
    "klimaks 1",
    "klimaks 2",
    "resolusi",
    "epilog"
]

bagian_instruction = {
    "orientasi 1": (
        "Perkenalkan tokoh utama, latar awal, dan situasi normal sebelum konflik. "
        "Bangun suasana cerita serta kenalkan elemen penting secara alami tanpa langsung menjelaskan semuanya."
    ),

    "orientasi 2": (
        "Perjelas tujuan, keinginan, atau motivasi tokoh utama. "
        "Perkenalkan petunjuk awal mengenai ancaman atau misteri yang akan memicu konflik."
    ),

    "konflik awal 1": (
        "Mulailah konflik utama dengan sebuah kejadian penting yang mengubah keadaan tokoh. "
        "Konflik harus nyata dan memaksa tokoh untuk bertindak."
    ),

    "konflik awal 2": (
        "Perlihatkan dampak konflik terhadap tokoh. "
        "Masalah menjadi semakin sulit dan pilihan tokoh mulai memiliki konsekuensi."
    ),

    "eskalasi 1": (
        "Perbesar konflik dengan menghadirkan rintangan, ancaman, atau hambatan baru. "
        "Keadaan harus lebih buruk daripada sebelumnya sehingga ketegangan meningkat."
    ),

    "eskalasi 2": (
        "Tekan tokoh hingga berada dalam situasi yang sangat sulit. "
        "Perlihatkan bahwa hampir semua jalan keluar tampak gagal sehingga tokoh harus membuat keputusan penting."
    ),

    "klimaks 1": (
        "Mulailah konfrontasi utama. "
        "Tokoh menghadapi sumber konflik secara langsung dan tidak bisa lagi menghindarinya."
    ),

    "klimaks 2": (
        "Capai puncak cerita. "
        "Tokoh mengambil keputusan atau tindakan paling penting yang menentukan hasil akhir cerita. "
        "Keputusan tersebut harus mengubah jalannya cerita."
    ),

    "resolusi": (
        "Perlihatkan akibat dari keputusan pada klimaks. "
        "Konflik utama mulai terselesaikan dan jelaskan bagaimana keadaan tokoh maupun dunia berubah."
    ),

    "epilog": (
        "Tunjukkan keadaan akhir tokoh setelah semua konflik selesai. "
        "Berikan penutup yang memberikan kesan utuh, bermakna, dan menjawab konflik utama tanpa membuka konflik baru."
    )
}
deployment_name = "qwen--qwen3-4b-instruct-2507"
# =========================
#  Bersihkan karakter
# =========================
import re

def clean_story(text: str) -> str:
    # Ganti karakter dengan spasi
    text = re.sub(r"[——–:;]", " ", text)

    # Rapikan spasi berlebih
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================
# Konfigurasi generate paragraf
# =========================
import time

def generate_paragraph(prompt):

    start = time.time()
    response = client.chat.completions.create(
        model= deployment_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9,
        top_p=0.95,
        max_tokens=10000
    )


    print(f"Generate: {time.time()-start:.2f} detik")

    result = response.choices[0].message.content

    if result is None:
        return ""

    result = clean_story(result)

    return result


# =========================
# Generate outline
# =========================
def generate_outline_ai(prompt):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": """
Anda adalah penulis novel profesional yang bertugas membuat outline cerita.

Ikuti informasi yang diberikan pengguna dan struktur cerita yang diberikan.
Pertahankan informasi penting dari pengguna.
Jika informasi belum lengkap, kembangkan secara kreatif dan masuk akal.
Pastikan alur cerita memiliki hubungan sebab-akibat yang jelas dan konsisten.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=1000
    )

    result = response.choices[0].message.content

    if result is None:
        return ""

    return result.strip()

def generate_outline(user_idea):

    prompt = f"""
Buat outline cerita berdasarkan ide atau instruksi pengguna berikut:
IDE PENGGUNA:
{user_idea}
Struktur cerita:

1. Orientasi 1
{bagian_instruction["orientasi 1"]}

2. Orientasi 2
{bagian_instruction["orientasi 2"]}

3. Konflik Awal 1
{bagian_instruction["konflik awal 1"]}

4. Konflik Awal 2
{bagian_instruction["konflik awal 2"]}

5. Eskalasi 1
{bagian_instruction["eskalasi 1"]}

6. Eskalasi 2
{bagian_instruction["eskalasi 2"]}

7. Klimaks 1
{bagian_instruction["klimaks 1"]}

8. Klimaks 2
{bagian_instruction["klimaks 2"]}

9. Resolusi
{bagian_instruction["resolusi"]}

10. Epilog
{bagian_instruction["epilog"]}

Aturan:

- Buat tepat 10 poin.
- Setiap poin maksimal 20 kata.
- Setiap poin hanya menjelaskan tujuan kejadian.
- Jangan menulis paragraf cerita.
"""
    return generate_outline_ai(prompt)

def generate_story(user_idea):

    # ====================
    # Membuat Outline
    # ====================

    outline = generate_outline(user_idea)

    print(outline)

    outline_list = []

    for line in outline.split("\n"):

        line = line.strip()

        if line.startswith(tuple(f"{i}." for i in range(1,11))):
            outline_list.append(line)

    if len(outline_list) != 10:
        print("Outline gagal dibuat")
        outline_list = [""] * 10

    paragraphs = []

    for i in range(10):

        bagian = bagian_list[i]
        outline_bagian = outline_list[i]

        # Ambil maksimal 3 paragraf sebelumnya
        if paragraphs:
            konteks = "\n\n".join(paragraphs[-3:])
        else:
            konteks = "(Belum ada paragraf sebelumnya)"

        prompt = f"""
Anda adalah penulis novel profesional.

Tugas Anda adalah menulis SATU paragraf cerita
berbahasa Indonesia berdasarkan ide, outline,
dan konteks cerita yang diberikan.

========================
IDE PENGGUNA
========================

{user_idea}

========================
KONTEKS CERITA SEBELUMNYA
========================

{konteks}

========================
BAGIAN CERITA
========================

{bagian}

Tujuan bagian:
{bagian_instruction[bagian]}

Outline bagian:
{outline_bagian}

========================
ATURAN PENULISAN
========================

1. Tulis maksimal 80 kata.
2. Gunakan bahasa Indonesia yang alami dan mengalir seperti novel Indonesia modern.
3. Fokus pada satu kejadian utama.
4. Hubungkan kejadian dengan konteks sebelumnya secara logis.
5. Pertahankan semua informasi penting dari ide pengguna.
6. Jangan mengubah identitas, tujuan, atau karakter tokoh yang sudah ditetapkan.
7. Jika informasi tertentu belum diberikan, kembangkan secara kreatif dan masuk akal.
8. Gunakan elemen penting dalam cerita secara nyata, bukan sekadar menyebutkannya.
9. Jangan menyelesaikan konflik utama sebelum bagian resolusi.
10. Akhiri paragraf dengan perkembangan yang mendorong cerita menuju bagian berikutnya.
11. Variasikan pembukaan paragraf.
12. Hindari pengulangan frasa, deskripsi, dialog, atau tindakan dari paragraf sebelumnya.
13. Jangan membuat paragraf menjadi rangkuman atau penjelasan.

========================
KONTINUITAS
========================

- Semua tindakan harus memiliki alasan yang jelas.
- Keputusan tokoh harus sesuai dengan situasi dan perkembangan cerita.
- Jangan tiba-tiba memunculkan tokoh penting tanpa alasan.
- Jangan tiba-tiba memunculkan benda penting tanpa penjelasan.
- Jangan berpindah lokasi secara tiba-tiba.
- Jangan mengubah tujuan tokoh tanpa perkembangan cerita.
- Jangan melupakan konflik dan tujuan utama cerita.
- Peristiwa baru harus memiliki hubungan dengan peristiwa sebelumnya.
- Perkembangan konflik harus mengikuti outline.

========================
PEMERIKSAAN
========================

Sebelum memberikan jawaban, periksa secara internal:

- Apakah paragraf terhubung dengan cerita sebelumnya?
- Apakah ada tokoh atau benda yang muncul tanpa alasan?
- Apakah ada informasi yang berulang?
- Apakah cerita bergerak maju?
- Apakah paragraf sesuai dengan tujuan bagian ini?

Jika ditemukan masalah, perbaiki sebelum memberikan jawaban.

Hanya tulis isi cerita.
Jangan memberikan penjelasan, analisis, atau komentar.
"""

        # ============================
        # Generate Paragraf
        # ============================

        paragraph = generate_paragraph(prompt)

        # Jika gagal
        if not paragraph:
            paragraph = "Cerita tidak dapat dilanjutkan."

        # Jika terlalu pendek, generate ulang
        elif len(paragraph.split()) < 30:
            paragraph = generate_paragraph(prompt)

        paragraphs.append(paragraph)

        print(f"Bagian {i + 1}: {bagian}")
        print(paragraph)
        print()

    # ============================
    # Gabungkan Cerita
    # ============================

    return "\n\n".join(paragraphs)

def send_whatsapp_message(to: str, message: str):

    url = (
        f"https://graph.facebook.com/v23.0/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )

    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": message
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    print("================================")
    print("WHATSAPP SEND RESPONSE")
    print("================================")
    print("Status:", response.status_code)
    print("Response:", response.text)

    return response
# =========================
# Endpoint
# =========================


@app.get("/webhook")
async def verify_webhook(request: Request):

    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(challenge)

    return PlainTextResponse(
        "Verification failed",
        status_code=403
    )

@app.post("/webhook")
async def receive_webhook(request: Request):

    data = await request.json()

    print("================================")
    print("WEBHOOK WHATSAPP MASUK")
    print("================================")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    try:

        value = data["entry"][0]["changes"][0]["value"]

        messages = value.get("messages")

        # Event bukan pesan masuk
        if not messages:
            return {"status": "ok"}

        message = messages[0]

        # Untuk sementara hanya menerima pesan text
        if message.get("type") != "text":
            return {"status": "ok"}

        sender = message["from"]
        text = message["text"]["body"]

        print("================================")
        print("PESAN WHATSAPP")
        print("================================")
        print("Pengirim:", sender)
        print("Pesan:", text)

        # TEST BALASAN
        send_whatsapp_message(
            sender,
            f"Pesan kamu diterima: {text}"
        )

        return {"status": "ok"}

    except Exception as e:

        print("================================")
        print("ERROR WEBHOOK")
        print("================================")

        import traceback
        traceback.print_exc()

        return {"status": "error"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    try:
        story = generate_story(req.message)

        return ChatResponse(reply=story)

    except Exception as e:
        import traceback
        traceback.print_exc()

        return ChatResponse(reply=str(e))
    
@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy():
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Kebijakan Privasi - Chatbot Pembuat Cerita</title>
    </head>
    <body>
        <h1>Kebijakan Privasi</h1>

        <p>
            Chatbot Pembuat Cerita menghargai privasi pengguna.
        </p>

        <h2>Data yang Dikumpulkan</h2>
        <p>
            Aplikasi dapat menerima pesan yang dikirim pengguna melalui
            WhatsApp untuk memproses permintaan pembuatan cerita.
        </p>

        <h2>Penggunaan Data</h2>
        <p>
            Data yang diterima digunakan untuk memproses permintaan pengguna
            dan menghasilkan respons cerita.
        </p>

        <h2>Penyimpanan Data</h2>
        <p>
            Aplikasi tidak menggunakan data pengguna untuk tujuan selain
            penyediaan layanan, kecuali sebagaimana diperlukan untuk
            menjalankan dan menjaga keamanan layanan.
        </p>

        <h2>Kontak</h2>
        <p>
            Untuk pertanyaan mengenai privasi, silakan hubungi:
            agelfikri12@gmail.com
        </p>
    </body>
    </html>
    """
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main3:app", host="0.0.0.0", port=8000, reload=True)