from ultralytics import YOLO
import cv2
import threading
import time

class ComputerVisionService:
    def __init__(self) -> None:
        self._trained = False
        self._is_running = False
        self._camera_thread = None
        self.cap = None
    
    def train_model(self) -> None:
        """giving the model a dataset to train, and be cappable of detect objects"""
        self.model.train(data="coco8.yaml", epochs=15, imgsz=640)
        self._trained = True
        
    

    def start_model(self) -> YOLO:
        """
        Starting the computer vision model to detect objects based on  the webcam of the computer
        """
        self.model = YOLO("yolov8n.pt")
        self.train_model()
        return self.model

    def activate_detection(self) -> dict:
        """
        Ativa a detecção de objetos via webcam
        Returns: Status da operação
        """
        if self._is_running:
            return {"status": "error", "message": "Detection is already running"}
        
        if not hasattr(self, 'model'):
            self.start_model()
        
        self._is_running = True
        self._camera_thread = threading.Thread(target=self._run_detection_loop)
        self._camera_thread.daemon = True
        self._camera_thread.start()
        
        return {"status": "success", "message": "Object detection activated"}
    
    def destroy_detection(self) -> dict:
        """
        Para e destrói a detecção de objetos
        Returns: Status da operação
        """
        if not self._is_running:
            return {"status": "error", "message": "Detection is not running"}
        
        self._is_running = False
        
        # Aguarda a thread terminar
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2)
        
        # Libera recursos da câmera
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        return {"status": "success", "message": "Object detection destroyed"}
    
    def get_status(self) -> dict:
        """
        Retorna o status atual do serviço
        """
        return {
            "status": "running" if self._is_running else "stopped",
            "trained": self._trained,
            "model_loaded": hasattr(self, 'model')
        }
    
    def _run_detection_loop(self) -> None:
        """
        Loop interno para detecção de objetos (executado em thread separada)
        """
        # Inicializa a webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            self._is_running = False
            return
        
        print("Object detection started. Press 'q' to quit in the window.")
        
        while self._is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            results = self.model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow('YOLO Object Detection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._is_running = False
                break
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Object detection stopped")
