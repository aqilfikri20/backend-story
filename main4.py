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
deployment_name = "qwen--qwen3-4b-instruct-2507"


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

# =========================
# Generate 1 paragraf
# =========================
import time

def generate_paragraph(prompt):

    start = time.time()
    response = client.chat.completions.create(
        model="qwen--qwen3-4b-instruct-2507",
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

    return result

# =========================
# Generate full story
# =========================
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

    paragraphs = []

    for i in range(10):
        bagian = bagian_list[i]

        if i == 0:

            prompt = f"""
Anda adalah penulis novel profesional.

Tugas Anda adalah menulis SATU paragraf cerita maksimal 80 kata.

Informasi cerita:

Nama Tokoh : {tokoh}
Genre : {genre}
Peran : {peran}
Sifat : {sifat}
Elemen Penting : {", ".join(elemen)}
Bagian cerita : {bagian}

Tujuan paragraf :
{bagian_instruction[bagian]}

Aturan:
- Lanjutkan cerita berdasarkan CERITA SEBELUMNYA.
- cerita yang anda buat diparagraf ini maksimal 80 kata.
- Jangan mengulang kejadian yang sudah terjadi.
- Jangan mengulang kalimat dari paragraf sebelumnya.
- Gunakan seluruh informasi tokoh, genre, peran, sifat, dan elemen penting secara konsisten sepanjang cerita.
- Jangan membuat tokoh utama baru.
- Gunakan nama {tokoh} secara konsisten.
- Tokoh utama tetap menjadi pusat cerita.
- Konflik harus berkembang secara logis.
- Gunakan seluruh elemen penting secara alami.
- Setiap paragraf harus memperlihatkan perkembangan konflik yang nyata, bukan hanya memperkenalkan atau mengulang simbol.
- Objek penting (buku, tas, kunci, dan sebagainya) harus memiliki fungsi yang jelas dalam alur cerita.
- Klimaks harus berisi tindakan atau keputusan penting yang mengubah arah cerita.
- Resolusi harus menjelaskan akibat dari keputusan tokoh utama
"""

        else:

            konteks = "\n\n".join(paragraphs[-3:])

            prompt = f"""
Anda adalah penulis novel profesional.

Tugas Anda adalah MELANJUTKAN cerita berikut.

======================
CERITA SEBELUMNYA
======================
{konteks}
======================

Informasi cerita:

Nama Tokoh : {tokoh}
Genre : {genre}
Peran : {peran}
Sifat : {sifat}
Elemen Penting : {", ".join(elemen)}
Bagian cerita : {bagian}

Tujuan paragraf:
{bagian_instruction[bagian]}

Aturan:
- Lanjutkan cerita berdasarkan CERITA SEBELUMNYA.
- cerita yang anda buat diparagraf ini maksimal 80 kata.
- Jangan mengulang kejadian yang sudah terjadi.
- Jangan mengulang kalimat dari paragraf sebelumnya.
- Gunakan seluruh informasi tokoh, genre, peran, sifat, dan elemen penting secara konsisten sepanjang cerita.
- Gunakan nama {tokoh} secara konsisten.
- Tokoh utama tetap menjadi pusat cerita.
- Konflik harus berkembang secara logis.
- Gunakan seluruh elemen penting secara alami.
- Setiap paragraf harus memperlihatkan perkembangan konflik yang nyata, bukan hanya memperkenalkan atau mengulang simbol.
- Objek penting (buku, tas, kunci, dan sebagainya) harus memiliki fungsi yang jelas dalam alur cerita.
- Klimaks harus berisi tindakan atau keputusan penting yang mengubah arah cerita.
- Resolusi harus menjelaskan akibat dari keputusan tokoh utama
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
    uvicorn.run("main4:app", host="0.0.0.0", port=8000, reload=True)