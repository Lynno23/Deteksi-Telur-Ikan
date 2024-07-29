from streamlit_webrtc import webrtc_streamer
import streamlit as st
import cv2
import ffmpeg
import openvino as ov
import torch
from ultralytics import YOLO
import os


# Fungsi utama Streamlit
#model = YOLO('best.pt')

# Export the model
#model.export(format="openvino", dynamic=True, half=True)  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
model = YOLO("best_model_openvino_model/")

def select_camera():
    # Mencari jumlah kamera yang tersedia
    num_cameras = 2  # Maksimum dua kamera: index 0 dan 1

    # Memeriksa kamera eksternal (index 1)
    cap_external = cv2.VideoCapture(1)
    if cap_external.isOpened():
        print("Menggunakan kamera eksternal (index 1)")
        return cap_external
    
    # Jika tidak ada kamera eksternal, menggunakan kamera default (index 0)
    print("Kamera eksternal tidak ditemukan, menggunakan kamera default (index 0)")
    cap_default = cv2.VideoCapture(0)
    if cap_default.isOpened():
        return cap_default
    
    # Jika tidak ada kamera yang terdeteksi sama sekali
    raise RuntimeError("Tidak dapat mengakses kamera")

def main():
  menu=st.sidebar.radio("Pilih Mode Deteksi:", ["Gambar", "Video","Webcam"])
  #menu = st.sidebar.selectbox("Pilih Mode Deteksi", ["Image", "Video","Webcam"])
  if menu == "Gambar":
    st.title("Deteksi dan Counting Objek Telur Ikan dengan Gambar")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
      save_path = os.path.join('uploads', uploaded_file.name)
      st.write(uploaded_file.name)
      with open(save_path, "wb") as f:
          f.write(uploaded_file.getvalue())
      temp_output_path = '{}_out.jpg'.format(save_path)
      st.success(f"File {uploaded_file.name} berhasil diunggah ke /uploads")
      img = cv2.imread(save_path)
      height, width, channels = img.shape
      results = model(img)

      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = height / 1000.0
      font_color = (255, 255, 255)
      thickness = int(height / 500.0)
      counts = {}
      text = ""

      for result in results:
        boxes = result.boxes.cpu().numpy()
        # Calculate and store object sizes along with bounding box coordinates
        for box in boxes:
            r = box.xyxy[0].astype(int)
            # Draw bounding box
            cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            # Display Counting object
            cls = int(box.cls[0])
            if not cls in counts.keys():
                counts[cls] = 1
            else:
                counts[cls] += 1
            for key in counts.keys():
                text = f"{model.names[key]}: {str(counts[key])}"
                
      cv2.putText(img, text, (10, 100), font, font_scale, (0,0,0), thickness+10)
      cv2.putText(img, text, (10, 100), font, font_scale, font_color, thickness)
      cv2.imwrite(temp_output_path, img)

      # Display input and output side-by-side
      col1, col2 = st.columns(2)

      # Display the original image in the first column
      col1.header("Input Image")
      col1.image(save_path, caption="Uploaded Image", use_column_width=True)

      col2.header("Output Image")
      col2.image(temp_output_path, caption="Object Detection Result", use_column_width=True)

  elif menu == "Video":
    st.title("Deteksi dan Counting Objek Telur Ikan dengan Video")
    # Pilih file video
    uploaded_file = st.file_uploader("Pilih file video", type=["mp4"])

    if uploaded_file is not None:
      # Menyimpan file yang diunggah ke direktori 'uploads'
      save_path = os.path.join('uploads', uploaded_file.name)
      with open(save_path, "wb") as f:
          f.write(uploaded_file.getvalue())
      temp_output_path = '{}_out.mp4'.format(save_path)
      st.success(f"File {uploaded_file.name} berhasil diunggah ke /uploads")

      # Baca file video
      cap = cv2.VideoCapture(save_path)

      # Baca frame dari video
      ret, frame = cap.read()
      # Ambil informasi video (lebar, tinggi)
      width = int(cap.get(3))
      height = int(cap.get(4))
      fps = int(cap.get(5))

      # Konfigurasi untuk menyimpan video
      out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1
      font_color = (0, 255, 0)
      thickness = 1
      counts = {}

      while ret:
        results = model(frame)[0]

        # Move the initialization of text inside the loop
        text = ""

        for result in results:
            boxes = result.boxes.cpu().numpy()

            # Calculate and store object sizes along with bounding box coordinates
            for box in boxes:
                r = box.xyxy[0].astype(int)
                # Draw bounding box
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cls = int(box.cls[0])
                if not cls in counts.keys():
                    counts[cls] = 1
                else:
                    counts[cls] += 1
                for key in counts.keys():
                    text = f"{model.names[key]}: {str(counts[key])}"

        cv2.putText(frame, text, (10, 30), font, font_scale, font_color, thickness)
        out.write(frame)
        ret, frame = cap.read()
        counts = {}  # Menginisialisasi ulang counts

      # Tutup file video dan file hasil
      cap.release()
      out.release()

      temp_final_path = '{}_fixed.mp4'.format(temp_output_path)
      #os.system(f'ffmpeg -i "{temp_output_path}" -vcodec libx264 -f mp4 "{temp_final_path}"') #local
      (
        ffmpeg
        .input(temp_output_path)
        .output(temp_final_path)
        .run()
      )

      # Tampilkan video hasil deteksi
      outvideo = open(temp_final_path, 'rb')
      video_bytes = outvideo.read()
      st.video(video_bytes)
      st.download_button(
        label="Download Output",
        data=outvideo,
        file_name='hasil_output.mp4',
        mime='video/mp4',
      )
      # Hapus file sementara setelah ditampilkan
      #os.remove(temp_output_path)

  elif menu == "Webcam":
    st.title("Deteksi dan Counting Objek Telur Ikan dengan Webcam")
    
      webrtc_streamer(key="sample", rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    '''
    run_webcam = st.checkbox('Run Webcam')
    text=""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    thickness = 1
    counts = {}
    if run_webcam:
        cap=select_camera()
        stframe = st.empty()
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image")
                break
            # Perform detection
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of bounding box
                    cls = int(box.cls[0])
                    if not cls in counts.keys():
                        counts[cls] = 1
                    else:
                        counts[cls] += 1
                    for key in counts.keys():
                        text = f"{model.names[key]}: {str(counts[key])}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (10, 300), font, font_scale, font_color, thickness)
            # Display results on the frame with reduced size
            stframe.image(frame, channels="BGR", use_column_width=False, width=800)  # Width set to 600 pixels
            counts = {}  # Menginisialisasi ulang counts
        cap.release()
        '''

if __name__ == "__main__":
    main()
