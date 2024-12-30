# Video Processing License Plates

This project detects vehicles in a video, tracks them, and performs OCR on detected license plates.  
It uses:
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection.  
- [Sort](https://github.com/abewley/sort) for multi-object tracking.  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for license plate text detection.  

## Features

1. Vehicle detection (cars, buses, trucks, etc.)  
2. Plate detection and OCR text extraction  
3. Tracking across frames  
4. CSV export of recognized plates  

## How to Use

1. Install dependencies from requirements.txt  
2. Place your video at ./sample_video.mp4 (or update the path in main.py)  
3. Adjust your license plate model in main.py if needed  
4. Run:  ```python main.py```
5. Check test.csv for results  

## How to Run
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

- main.py: Main entry point (loads models, processes frames, exports CSV)  
- util.py: OCR functions, plate formatting, CSV handling  
- sort/: Contains the SORT algorithm code  
- models/: Where license_plate_detector.pt is stored  
- requirements.txt: Python dependencies  

## License

[MIT License](LICENSE)