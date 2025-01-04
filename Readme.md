# Automatic License Plate Detection

## **Project Overview**
This project focuses on **automatic license plate detection** using **YOLOv8 (You Only Look Once)**, leveraging **PyTorch**, **Python**, and **computer vision** techniques. The goal is to accurately detect and localize license plates in images for applications like traffic monitoring and vehicle identification.

---

## **Tech Stack**
- **Frameworks**: PyTorch, Ultralytics YOLOv8
- **Languages**: Python
- **Libraries**: OpenCV, Pandas, Matplotlib, NumPy, scikit-learn
- **Computer Vision Techniques**: Object detection, bounding box localization

---

## **Project Details**
1. **Dataset**:
   - Consists of **433 pre-annotated images**.
   - Annotations are provided in XML format (PASCAL VOC).

2. **Data Preprocessing**:
   - Parsed XML annotations and converted them into **YOLO-compatible format**.
   - Split data into **train (80%)**, **validation (10%)**, and **test (10%)** sets.

3. **Model Training**:
   - Trained a **YOLOv8n** model for **100 epochs** using GPU acceleration.
   - Utilized **batch size = 16** and **image size = 320x320**.
   - Cached images for faster training.

4. **Performance Metrics**:
   - **mAP@0.5 = 0.7**: Measures precision-recall performance at 50% IoU threshold.
   - **mAP@0.5:0.95 = 0.5**: Evaluates stricter localization performance.

5. **Deployment**:
   - Deployed the model using **Streamlit**, making it accessible via a web interface.

---

## **Folder Structure**
```
|-- datasets
    |-- train
        |-- images
        |-- labels
    |-- val
        |-- images
        |-- labels
    |-- test
        |-- images
        |-- labels
|-- models
|-- scripts
|-- datasets.yaml
|-- app.py
|-- requirements.txt
|-- README.md
```

---


---

## **Usage**
### **1. Training the Model**
```bash
python scripts/train.py --epochs 100 --batch 16 --imgsz 320
```

### **2. Testing the Model**
```bash
python scripts/test.py --weights best_license_plate_model.pt
```

### **3. Launching the Application**
```bash
streamlit run app.py
```

---

## **Results**
- **Precision**: 91.4%
- **Recall**: 86.4%
- **mAP@0.5**: 90.9%
- **mAP@0.5:0.95**: 55.6%

---

## **Future Improvements**
- Enhance performance at **higher IoU thresholds** by fine-tuning anchor boxes and augmenting training data.
- Optimize the model further for **mobile or embedded systems**.
- Integrate license plate **OCR (Optical Character Recognition)** for text extraction.

---

## **License**
This project is licensed under the **MIT License**.

---

---


