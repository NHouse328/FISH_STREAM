import cv2
import ffmpeg
import numpy as np

# Função para adicionar texto no vídeo
def add_text_to_frame(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Configurações do vídeo
input_url = 'rtmp://localhost'  # URL de entrada
output_url = 'rtmp://localhost/live/output'  # URL de saída
text_to_display = 'Seu Texto Aqui'  # O texto que você quer exibir

# Criação do processo de streaming com FFmpeg
process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='1920x1080')  # Ajuste a resolução conforme necessário
    .output(output_url, format='flv', pix_fmt='yuv420p')
    .overwrite_output()
    .global_args('-re')
    .run_async(pipe_stdin=True)
)

# Captura de vídeo do stream de entrada
cap = cv2.VideoCapture(input_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Adiciona texto ao frame
    add_text_to_frame(frame, text_to_display)

    # Converte o frame para o formato necessário e envia para o processo de streaming
    process.stdin.write(frame.tobytes())

cap.release()
process.stdin.close()
process.wait()
