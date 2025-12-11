import streamlit as st
import cv2
import numpy as np
import imutils
from imutils import contours
from imutils.perspective import four_point_transform

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Scanner LJK Universal", layout="wide")

# --- FUNGSI CORE ---
def pre_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge Detection otomatis
    edged = cv2.Canny(blurred, 75, 200)
    return edged, gray

def get_paper_contour(edged):
    # Cari kontur kertas (objek terbesar segiempat)
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

def scan_bubbles(warped_gray, min_w, max_w, min_h, max_h, num_questions, num_options):
    # 1. Thresholding (Hitam Putih)
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 2. Cari Semua Kontur (Calon Jawaban)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    questionCnts = []
    
    # 3. Filter Ukuran (Sesuai Slider User)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        # Filter berdasarkan ukuran pixel dan aspek rasio (mendekati kotak/bulat)
        if w >= min_w and w <= max_w and h >= min_h and h <= max_h and ar >= 0.7 and ar <= 1.3:
            questionCnts.append(c)
            
    # Visualisasi apa yang dideteksi
    debug_img = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, questionCnts, -1, (0, 255, 0), 2)
    
    total_expected = num_questions * num_options
    
    if len(questionCnts) != total_expected:
        return None, f"Ditemukan {len(questionCnts)} bulatan. Harusnya {total_expected}. Silakan atur 'Sensitivitas Ukuran' di sebelah kiri.", debug_img, thresh

    # 4. Sorting (PENTING: Logika ini untuk 1 Kolom memanjang ke bawah)
    # Jika LJK 2 kolom, logika sorting harus diubah (split list).
    # Untuk versi Universal Basic, kita asumsikan 1 blok pertanyaan utuh.
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    
    results = []
    score = 0
    
    # 5. Grading
    for (q, i) in enumerate(np.arange(0, len(questionCnts), num_options)):
        cnts_row = contours.sort_contours(questionCnts[i:i + num_options])[0]
        bubbled = None
        
        for (j, c) in enumerate(cnts_row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        
        # Simpan jawaban (Index 0=A, 1=B, dst)
        results.append(bubbled[1])
        
        # Visualisasi jawaban terdeteksi
        cv2.drawContours(debug_img, [cnts_row[bubbled[1]]], -1, (0, 0, 255), 3)

    return results, "Sukses", debug_img, thresh

# --- UI STREAMLIT ---
st.title("🛠️ Scanner LJK Universal (Configurable)")
st.markdown("""
Aplikasi ini bisa membaca LJK buatan sendiri asalkan:
1. Memiliki **4 Kotak Hitam** di sudut (Anchor).
2. Susunan jawaban **Rapi (Grid)**.
3. Anda mengatur **Jumlah Soal** dan **Ukuran Bulatan** di menu samping.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Konfigurasi LJK")
    st.info("Sesuaikan parameter ini dengan bentuk kertas Anda.")
    
    num_q = st.number_input("Jumlah Soal", min_value=1, max_value=100, value=10)
    num_opt = st.selectbox("Jumlah Opsi Jawaban", [3, 4, 5], index=2)
    
    st.write("---")
    st.write("**Kalibrasi Deteksi Bulatan**")
    st.write("Geser slider sampai kotak hijau pas membungkus bulatan jawaban.")
    min_w = st.slider("Lebar Minimal (px)", 10, 50, 20)
    max_w = st.slider("Lebar Maksimal (px)", 30, 100, 50)
    
    st.write("---")
    st.header("🔑 Kunci Jawaban")
    ans_key_str = st.text_area("Masukkan Kunci (Pisahkan koma, cth: A,B,C...)", "A,B,A,C,A,B,E,D,C,A")
    
    # Parsing Kunci
    try:
        ans_key = [ord(x.strip().upper()) - 65 for x in ans_key_str.split(',')]
        if len(ans_key) != num_q:
            st.warning(f"Jumlah kunci ({len(ans_key)}) tidak sama dengan Jumlah Soal ({num_q}).")
    except:
        st.error("Format kunci salah.")
        ans_key = []

with col2:
    st.header("📸 Upload & Scan")
    uploaded_file = st.file_uploader("Upload Foto LJK Siswa", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # 1. Cari Kertas
        edged, gray = pre_process(image)
        docCnt = get_paper_contour(edged)
        
        if docCnt is None:
            st.error("❌ Gagal mendeteksi kertas. Pastikan foto di alas gelap & 4 sudut terlihat.")
            st.image(image, width=300)
        else:
            # Warp Perspective
            paper = four_point_transform(image, docCnt.reshape(4, 2))
            warped_gray = four_point_transform(gray, docCnt.reshape(4, 2))
            
            # SCAN
            st.write("Sedang menganalisis struktur...")
            result_data, msg, debug_img, thresh_img = scan_bubbles(
                warped_gray, min_w, max_w, min_w, max_w, num_q, num_opt
            )
            
            # TABS Visualisasi
            tab1, tab2, tab3 = st.tabs(["Hasil Akhir", "Debug Deteksi", "Mata Komputer (B/W)"])
            
            with tab2:
                st.image(debug_img, caption="Kotak Hijau = Terdeteksi sebagai Soal", channels="BGR")
                if result_data is None:
                    st.error(msg)
            
            with tab3:
                st.image(thresh_img, caption="Penglihatan Komputer (Threshold)", channels="GRAY")

            with tab1:
                if result_data is not None:
                    # Hitung Nilai
                    correct = 0
                    detail_res = []
                    for i, ans in enumerate(result_data):
                        # Cek bounds
                        kunci = ans_key[i] if i < len(ans_key) else -1
                        status = "✅" if ans == kunci else "❌"
                        if ans == kunci: correct += 1
                        
                        huruf_siswa = chr(65 + ans)
                        huruf_kunci = chr(65 + kunci) if kunci >= 0 else "?"
                        detail_res.append(f"No {i+1}: {huruf_siswa} (Kunci: {huruf_kunci}) {status}")
                    
                    final_score = (correct / len(ans_key)) * 100 if len(ans_key) > 0 else 0
                    
                    st.balloons()
                    st.success(f"### NILAI: {final_score:.2f}")
                    st.progress(final_score / 100)
                    
                    with st.expander("Lihat Detail Jawaban"):
                        st.write(detail_res)
                else:
                    st.warning("⚠️ Hasil scan belum muncul karena jumlah bulatan tidak pas. Cek Tab 'Debug Deteksi' dan atur Slider di kiri.")
