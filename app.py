# File: app_scanner.py
import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

# --- LOGIKA PEMROSESAN CITRA ---

def pre_process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged, gray

def find_document_corners(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt

def process_exam(image, answer_key):
    # 1. Deteksi Kertas
    edged, gray = pre_process_image(image)
    docCnt = find_document_corners(edged)

    if docCnt is None:
        return None, "Gagal mendeteksi sudut kertas. Pastikan foto jelas dan background kontras."

    # 2. Luruskan Perspektif (Warp)
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped_gray = four_point_transform(gray, docCnt.reshape(4, 2))

    # 3. Thresholding & Morfologi (Khusus Tanda Silang)
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Teknik DILATION: Menebalkan garis 'X' agar terbaca
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 4. Cari Kotak Jawaban
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # Filter kontur yang menyerupai kotak jawaban
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.2:
            questionCnts.append(c)

    # Urutkan kontur (Top-to-Bottom)
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

    # Validasi jumlah kotak (Harus 50 kotak: 10 soal x 5 pilihan)
    if len(questionCnts) != 50:
        return None, f"Terdeteksi {len(questionCnts)} kotak. Harusnya 50. Cek pencahayaan/bayangan."

    # 5. Penilaian
    correct = 0
    results = [] # Simpan detail per soal

    # Loop per baris (5 kotak per soal)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts_row = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None
        
        # Cek mana yang disilang
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            
            # Inner Crop logic (buang pinggiran kotak)
            x, y, w, h = cv2.boundingRect(c)
            # Hitung pixel hanya di area mask
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            # Jika pixel putih cukup banyak (Treshold sensitivitas X)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # Cek Jawaban Benar/Salah
        k = answer_key[q]
        user_answer_idx = bubbled[1]
        
        # Validasi "Kosong" (Threshold Noise)
        # Jika pixel terdeteksi sangat sedikit, anggap tidak dijawab
        if bubbled[0] < 100: # Angka ini perlu tuning tergantung resolusi kamera
             user_answer_idx = -1 # Tidak dijawab

        color = (0, 0, 255) # Merah (Salah)
        status = "SALAH"

        if k == user_answer_idx:
            color = (0, 255, 0) # Hijau (Benar)
            correct += 1
            status = "BENAR"
        elif user_answer_idx == -1:
            status = "KOSONG"
            color = (0, 255, 255) # Kuning

        results.append(f"Soal {q+1}: Jawab {chr(65+user_answer_idx) if user_answer_idx >=0 else '-'} | Kunci {chr(65+k)} -> {status}")

        # Visualisasi Hasil di Gambar
        cv2.drawContours(paper, [cnts_row[k]], -1, (255, 255, 255), 3) # Lingkari Kunci Jawaban (Putih)
        if user_answer_idx != -1:
            cv2.drawContours(paper, [cnts_row[user_answer_idx]], -1, color, 3) # Lingkari Jawaban Siswa

    score = (correct / 10) * 100
    return (score, paper, results), "Sukses"

# --- USER INTERFACE (STREAMLIT) ---

st.title("📱 Scanner LJK Pintar (Support Tanda 'X')")
st.write("Upload foto LJK yang sudah diisi. Pastikan 4 sudut kotak hitam terlihat jelas.")

# 1. Input Kunci Jawaban
st.sidebar.header("Kunci Jawaban")
answer_key_indices = {}
options = ['A', 'B', 'C', 'D', 'E']
for i in range(10):
    ans = st.sidebar.selectbox(f"Soal {i+1}", options, index=0, key=i)
    answer_key_indices[i] = options.index(ans)

# 2. Upload Gambar
uploaded_file = st.file_uploader("Pilih Foto LJK...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file upload ke OpenCV Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption='Gambar Asli', use_column_width=True)

    if st.button("🔍 SCAN SEKARANG"):
        with st.spinner('Sedang memindai jawaban siswa...'):
            data, message = process_exam(image, answer_key_indices)

            if data is None:
                st.error(f"Error: {message}")
            else:
                score, result_img, details = data
                
                # Tampilkan Skor
                st.success("Selesai!")
                st.metric(label="NILAI AKHIR", value=f"{score:.1f}")

                # Tampilkan Gambar Hasil Koreksi
                st.image(result_img, caption='Hasil Koreksi (Hijau=Benar, Merah=Salah, Putih=Kunci)', use_column_width=True, channels="BGR")

                # Tampilkan Rincian
                with st.expander("Lihat Rincian Jawaban"):
                    for det in details:
                        st.write(det)
