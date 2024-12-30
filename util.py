import string
import easyocr
import csv

# Inicializa o OCR (ex.: modo GPU True se disponível)
ocr_reader = easyocr.Reader(['en'], gpu=True)

# Dicionários de mapeamento para conversão de caracteres
char_to_digit_map = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5'
}

digit_to_char_map = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S'
}


def export_detections_to_csv(detection_results, csv_path):
    """
    Exporta os resultados de detecção para um arquivo CSV.
    
    Args:
        detection_results (dict): Dicionário contendo todos os resultados de detecção por frame.
        csv_path (str): Caminho onde será salvo o arquivo CSV.
    """
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_number', 'vehicle_id', 'license_number'])

        for frame_num, vehicles in detection_results.items():
            for vehicle_id, data in vehicles.items():
                writer.writerow([
                    data['frame_number'],
                    data['vehicle_id'],
                    data['license_number']
                ])


def is_valid_license_plate_format(text):
    """
    Verifica se a string tem 7 caracteres e segue o padrão básico de placa (L L D D L L L),
    levando em conta possíveis conversões O->0, I->1 etc.

    Args:
        text (str): Texto a ser verificado.

    Returns:
        bool: True se tiver 7 caracteres e for plausível, False caso contrário.
    """
    if len(text) != 7:
        return False

    # L = letra (A-Z ou mapeável)
    # D = dígito (0-9 ou mapeável)
    # Formato: L L D D L L L
    if (text[0] in string.ascii_uppercase or text[0] in digit_to_char_map) and \
       (text[1] in string.ascii_uppercase or text[1] in digit_to_char_map) and \
       (text[2] in string.digits or text[2] in char_to_digit_map) and \
       (text[3] in string.digits or text[3] in char_to_digit_map) and \
       (text[4] in string.ascii_uppercase or text[4] in digit_to_char_map) and \
       (text[5] in string.ascii_uppercase or text[5] in digit_to_char_map) and \
       (text[6] in string.ascii_uppercase or text[6] in digit_to_char_map):
        return True

    return False



def format_license_plate(text):
    """
    Aplica uma formatação na string de placa, utilizando mapeamentos diferentes
    para cada posição (ex.: conversão dígito->caractere ou caractere->dígito).

    Args:
        text (str): Placa bruta reconhecida.

    Returns:
        str: Placa formatada segundo os dicionários digit_to_char_map e char_to_digit_map.
    """
    license_output = ''
    # Posição 0,1,4,5,6 -> digit_to_char_map
    # Posição 2,3       -> char_to_digit_map
    position_mapping = {
        0: digit_to_char_map,
        1: digit_to_char_map,
        4: digit_to_char_map,
        5: digit_to_char_map,
        6: digit_to_char_map,
        2: char_to_digit_map,
        3: char_to_digit_map
    }

    for j in range(7):
        if text[j] in position_mapping[j]:
            license_output += position_mapping[j][text[j]]
        else:
            license_output += text[j]

    return license_output


def extract_license_plate_text(plate_crop):
    """
    Extrai (via OCR) o texto de placa a partir de uma imagem recortada.

    Args:
        plate_crop (np.array): Recorte (cropped) da placa em escala de cinza ou BGR.

    Returns:
        (str | None, float | None): Texto da placa (string) e confiança.
                                    Se não encontrado, retorna (None, None).
    """
    detections = ocr_reader.readtext(plate_crop)
    for detection in detections:
        _, text, score = detection
        text = text.upper().replace(' ', '')

        # Verifica se atende ao formato básico
        if is_valid_license_plate_format(text):
            # Verifica se a placa pode ser interpretada como um formato brasileiro
            valid_br = format_license_plate(text)
            return valid_br, score

    return None, None


def locate_tracked_vehicle(license_plate_box, tracked_vehicles):
    """
    Localiza o veículo ao qual a placa pertence, verificando se o bounding box
    da placa está contido dentro do bounding box do veículo.

    Args:
        license_plate_box (list|tuple): [x1, y1, x2, y2, score, class_id] da placa.
        tracked_vehicles (ndarray|list): Lista/array de veículos rastreados ([vx1, vy1, vx2, vy2, id]).

    Returns:
        tuple:
            (vx1, vy1, vx2, vy2, vehicle_id) se encontrado,
            ou (-1, -1, -1, -1, -1) caso não encontre.
    """
    x1, y1, x2, y2, _, _ = license_plate_box

    found_vehicle = False
    vehicle_index = -1
    for idx, (vx1, vy1, vx2, vy2, vehicle_id) in enumerate(tracked_vehicles):
        # Verifica se os limites da placa estão dentro do bounding box do veículo
        if x1 > vx1 and y1 > vy1 and x2 < vx2 and y2 < vy2:
            vehicle_index = idx
            found_vehicle = True
            break

    if found_vehicle:
        return tracked_vehicles[vehicle_index]

    return -1, -1, -1, -1, -1
