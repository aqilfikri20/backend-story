# main.py
from urllib import response
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import random
import json
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AZURE_API_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

# =========================
# Schema
# =========================
class StoryRequest(BaseModel):
    nama: str = ""
    genre: str = ""
    sifat: str = ""
    peran: str = ""
    elemen: List[str] = []

class ChatResponse(BaseModel):
    reply: str

# =========================
# Load relations
# =========================
with open("relations.json", "r", encoding="utf-8") as f:
    relations = json.load(f)

def weighted_pick(data):
    if not data:
        return None

    return random.choices(
        population=list(data.keys()),
        weights=list(data.values()),
        k=1
    )[0]

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
deployment_name = "qwen--qwen3-4b-instruct-2507-2"
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
# Generate full story
# =========================
def generate_outline_ai(prompt):

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.5,
        top_p=0.9,
        max_tokens=1000
    )

    return response.choices[0].message.content

def generate_outline(tokoh, genre, peran, sifat, elemen):

    prompt = f"""
Anda adalah penulis novel profesional.
Buat outline cerita menjadi tepat 10 bagian.

Informasi cerita
Nama Tokoh : {tokoh}
Genre : {genre}
Peran : {peran}
Sifat : {sifat}
Elemen Penting : {", ".join(elemen)}

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

def generate_story(data: StoryRequest):

    tokoh = data.nama.strip()
    genre = data.genre.strip()
    peran = data.peran.strip()

    sifat = data.sifat.strip()

    elemen = [
        e.strip()
        for e in data.elemen
        if e.strip()
    ]

    if not elemen:
        elemen.append("Petualangan")

    # ====================
    # Membuat Outline
    # ====================

    outline = generate_outline(
        tokoh,
        genre,
        peran,
        sifat,
        elemen
    )

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

        if i == 0:

            prompt = f"""
Anda adalah penulis novel profesional.
Tugas Anda adalah menulis SATU paragraf cerita berbahasa Indonesia.
Informasi cerita

Nama Tokoh : {tokoh}
Genre : {genre}
Peran : {peran}
Sifat : {sifat}
Elemen Penting : {", ".join(elemen)}
Bagian Cerita : {bagian}
Tujuan paragraf:
{bagian_instruction[bagian]}
Outline bagian ini:
{outline_bagian}

Aturan penulisan:
1. Tulis maksimal 80 kata.
2. Gunakan bahasa Indonesia yang alami, mudah dipahami, dan mengalir seperti novel Indonesia modern.
3. Fokus pada satu kejadian utama dalam paragraf ini.
4. Paragraf harus memiliki hubungan sebab-akibat yang jelas dengan paragraf sebelumnya.
5. Tokoh utama adalah {tokoh}. Jangan membuat tokoh utama baru kecuali benar-benar diperlukan.
6. Gunakan sifat "{sifat}" melalui tindakan, dialog, dan keputusan tokoh, bukan dengan menyebutkannya secara langsung.
7. Elemen penting harus memiliki fungsi nyata dalam cerita, bukan hanya disebutkan.
8. Akhiri paragraf dengan situasi yang mendorong cerita berlanjut secara alami.
9. Hanya tulis isi cerita tanpa penjelasan.
10. Variasikan pembukaan paragraf.
11.Jangan selalu memulai paragraf dengan nama tokoh atau tindakan yang sama.
12. Hindari mengulang frasa, deskripsi tubuh, maupun dialog yang sama di beberapa paragraf.

Jangan:
- Jangan membuat kejadian yang melompat terlalu jauh.
- Jangan memunculkan benda baru tanpa penjelasan.
- Jangan tiba-tiba berpindah lokasi.
- Jangan menyelesaikan konflik terlalu cepat.
- Jangan memperkenalkan musuh baru jika konflik lama belum selesai.

Kontinuitas cerita:
- Semua tindakan harus merupakan akibat logis dari paragraf sebelumnya.
- Semua keputusan tokoh harus memiliki alasan.
- Semua benda penting yang muncul harus tetap diingat sampai cerita selesai.
- Jangan melupakan konflik utama.
- Jangan melupakan tujuan tokoh.

Periksa paragraf di atas.
Jawab:
1. Apakah ada benda yang muncul tiba-tiba?
2. Apakah ada tokoh yang berubah tujuan tanpa alasan?
3. Apakah ada kalimat yang mengulang informasi paragraf sebelumnya?
4. Apakah konflik bergerak maju?
Jika ada, tulis ulang paragraf tersebut menjadi lebih baik.
Hanya tulis isi cerita tanpa penjelasan.
"""

        else:

            konteks = "\n\n".join(paragraphs[-3:])

            prompt = f"""
Anda adalah penulis novel profesional.
Tugas Anda adalah MELANJUTKAN cerita berikut.

CERITA SEBELUMNYA
{konteks}

Informasi cerita:
Nama Tokoh : {tokoh}
Genre : {genre}
Peran : {peran}
Sifat : {sifat}
Elemen Penting : {", ".join(elemen)}
Bagian cerita : {bagian}
Tujuan paragraf:
{bagian_instruction[bagian]}
Outline bagian ini:
{outline_bagian}

Aturan penulisan:
1. Tulis maksimal 80 kata.
2. Gunakan bahasa Indonesia yang alami, mudah dipahami, dan mengalir seperti novel Indonesia modern.
3. Fokus pada satu kejadian utama dalam paragraf ini.
4. Paragraf harus memiliki hubungan sebab-akibat yang jelas dengan paragraf sebelumnya.
5. Tokoh utama adalah {tokoh}. Jangan membuat tokoh utama baru kecuali benar-benar diperlukan.
6. Gunakan sifat "{sifat}" melalui tindakan, dialog, dan keputusan tokoh, bukan dengan menyebutkannya secara langsung.
7. Elemen penting harus memiliki fungsi nyata dalam cerita, bukan hanya disebutkan.
8. Akhiri paragraf dengan situasi yang mendorong cerita berlanjut secara alami.
9. Hanya tulis isi cerita tanpa penjelasan.
10. Variasikan pembukaan paragraf.
11.Jangan selalu memulai paragraf dengan nama tokoh atau tindakan yang sama.
12. Hindari mengulang frasa, deskripsi tubuh, maupun dialog yang sama di beberapa paragraf.

Jangan:
- Jangan membuat kejadian yang melompat terlalu jauh.
- Jangan memunculkan benda baru tanpa penjelasan.
- Jangan tiba-tiba berpindah lokasi.
- Jangan menyelesaikan konflik terlalu cepat.
- Jangan memperkenalkan musuh baru jika konflik lama belum selesai.

Kontinuitas cerita:
- Semua tindakan harus merupakan akibat logis dari paragraf sebelumnya.
- Semua keputusan tokoh harus memiliki alasan.
- Semua benda penting yang muncul harus tetap diingat sampai cerita selesai.
- Jangan melupakan konflik utama.
- Jangan melupakan tujuan tokoh.

Periksa paragraf di atas.
Jawab:
1. Apakah ada benda yang muncul tiba-tiba?
2. Apakah ada tokoh yang berubah tujuan tanpa alasan?
3. Apakah ada kalimat yang mengulang informasi paragraf sebelumnya?
4. Apakah konflik bergerak maju?
Jika ada, tulis ulang paragraf tersebut menjadi lebih baik.
Hanya tulis isi cerita tanpa penjelasan.
"""

        paragraph = generate_paragraph(prompt)

        if not paragraph:
            paragraph = "Cerita tidak dapat dilanjutkan."
        elif len(paragraph.split()) < 30:
            paragraph = generate_paragraph(prompt)

        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)
# =========================
# Endpoint
# =========================
@app.options("/chat")
def chat_options():
    return JSONResponse(status_code=200, content={"message": "OK"})


@app.post("/chat", response_model=ChatResponse)
def chat(req: StoryRequest):

    try:
        story = generate_story(req)

        return ChatResponse(reply=story)

    except Exception as e:
        import traceback
        traceback.print_exc()

        return ChatResponse(reply=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=True)