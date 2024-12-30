import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

from util import (
    locate_tracked_vehicle,
    extract_license_plate_text,
    export_detections_to_csv
)


def load_models():
    """
    Carrega os modelos de detecção de veículos e de placas.
    
    Returns:
        YOLO: Modelo YOLO para detecção de veículos.
        YOLO: Modelo YOLO para detecção de placas.
    """
    vehicle_detector = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
    return vehicle_detector, license_plate_detector


def main():
    """
    Função principal que executa todo o pipeline:
    1. Carrega modelos.
    2. Abre o vídeo e itera frame a frame.
    3. Detecta veículos e faz tracking.
    4. Detecta placas, recorta e reconhece o texto.
    5. Salva os resultados em CSV.
    """
    # Carrega os modelos
    vehicle_detector, license_plate_detector = load_models()

    # Cria objeto de captura de vídeo
    video_capture = cv2.VideoCapture('./sample_video.mp4')

    # Cria o rastreador de múltiplos objetos
    multi_object_tracker = Sort()

    # Dicionário para armazenar resultados de detecção e OCR
    detection_results = {}

    # Classes numéricas correspondentes a veículos (ex.: carro, moto, ônibus, caminhão)
    vehicle_classes = [2, 3, 5, 7]

    frame_number = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Inicializa a estrutura de resultados para o frame corrente
        detection_results[frame_number] = {}

        # ========================
        # 1) Detecção de veículos
        # ========================
        vehicle_detections = vehicle_detector(frame)[0]
        # Filtra apenas as detecções de veículos
        detection_boxes = []
        for box in vehicle_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) in vehicle_classes:
                detection_boxes.append([x1, y1, x2, y2, score])

        # ========================
        # 2) Rastreamento (Tracking)
        # ========================
        track_ids = multi_object_tracker.update(np.asarray(detection_boxes))

        # ========================
        # 3) Detecção de placas
        # ========================
        plate_detections = license_plate_detector(frame)[0]
        for plate_box in plate_detections.boxes.data.tolist():
            px1, py1, px2, py2, plate_score, plate_class_id = plate_box

            # Associa a placa detectada a um veículo rastreado
            xcar1, ycar1, xcar2, ycar2, vehicle_id = locate_tracked_vehicle(plate_box, track_ids)
            if vehicle_id != -1:
                # Recorta a área da placa
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]
                plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, plate_crop_thresh = cv2.threshold(plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Reconhece (OCR) o texto da placa
                license_text, license_confidence = extract_license_plate_text(plate_crop_thresh)
                if license_text is not None:
                    detection_results[frame_number][vehicle_id] = {
                        'frame_number': frame_number,
                        'vehicle_id': vehicle_id,
                        'license_number': license_text
                    }

        frame_number += 1

    # ========================
    # 4) Salva resultados em CSV
    # ========================
    export_detections_to_csv(detection_results, './test.csv')


if __name__ == '__main__':
    main()
