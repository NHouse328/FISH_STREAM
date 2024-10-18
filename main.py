import cv2
import torch
import subprocess
from torchvision import transforms
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Carregar modelo YOLOv5
model_path = 'yolov5s.pt'  # Usando o modelo pré-treinado 'small'
device = select_device('cpu')  # Troque para 'cuda' se tiver uma GPU
model = DetectMultiBackend(model_path, device=device)

# Função para detectar objetos no frame
def detect_objects(frame):
    resize_transform = transforms.Resize((1920, 1080))
    img = resize_transform(frame)

    # Converte a imagem para tensor
    img = torch.from_numpy(img).float().to(device)
    img = img.permute(2, 0, 1).unsqueeze(0)  # Muda a ordem das dimensões

    # Realiza a detecção
    pred = model(img)

    # Aplicar não máxima supressão
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    return pred

# Função para desenhar boxes e labels no frame
def draw_boxes(frame, detections):
    for det in detections:
        if len(det):
            for *box, conf, cls in det:
                x1, y1, x2, y2 = map(int, box)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# RTMP input and output settings
input_rtmp = 'rtmp://localhost'  # URL de entrada RTMP
output_rtmp = 'rtmp://localhost/live/output'  # URL de saída RTMP

# Captura de vídeo do RTMP
cap = cv2.VideoCapture(input_rtmp)

# Verificar se a captura foi iniciada
if not cap.isOpened():
    print("Erro ao abrir o fluxo RTMP.")
    exit()

# Comando FFmpeg para transmitir o vídeo processado via RTMP
ffmpeg_process = subprocess.Popen([ 
    'ffmpeg', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1920x1080', 
    '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-f', 'flv', output_rtmp 
], stdin=subprocess.PIPE)

# Loop de processamento
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos
    detections = detect_objects(frame)

    # Desenhar boxes e labels
    frame = draw_boxes(frame, detections)

    # Redimensionar o frame para 1920x1080 para a transmissão
    frame_resized = cv2.resize(frame, (1920, 1080))

    # Transmitir o vídeo de volta via RTMP
    ffmpeg_process.stdin.write(frame_resized.tobytes())

# Liberar recursos
cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
