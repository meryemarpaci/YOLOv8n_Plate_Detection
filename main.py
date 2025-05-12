# YoloV8
from ultralytics import YOLO
# yolov8n => bir computer vision modeli
import os
import cv2
import matplotlib.pyplot as plt

def train():
    model = YOLO("yolov8n.pt")
    # Daha hızlı eğitim için epochs değerini 3'e düşürdüm
    model.train(data="data.yaml", epochs=10, imgsz=320, batch=4, patience=5)

# Pythonda starndart koruma yapısı.
# python {main.py} => bu kodu çalıştırır.
# ama diğer dosyalar import ettiğinde burayı çalıştırmaz.
#if __name__ == '__main__':
#    train()

####
#full dataset ile train
def train_full_dataset():
    # Önceden eğitilmiş yolo modeli
    model = YOLO("yolov8n.pt")
    
    # epochs:10, imgsz (görüntü boyutu): 320x320, batch:16
    results = model.train(data="data.yaml", epochs=10, imgsz=320, batch=16, patience=5)
    
    print(f"Eğitim tamamlandı! Model otomatik olarak runs/detect/train klasörüne kaydedildi.")
    return model

# Modeli kaydetme ve yeni bir resimle test etme fonsiyonum
def save_and_test_model():
    model = train_full_dataset()
    
    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/plaka_tanima_model.pt"
    model.export(format="onnx")  # ONNX formatında dışa aktarma
    
    best_model_path = os.path.join(os.getcwd(), "runs/detect/train/weights/best.pt")
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, model_path)
        print(f"En iyi model {model_path} konumuna kaydedildi.")
    
    # Yeni bir görüntüyle test etme
    test_image = "deep-learning/plate/dataset/images/3.jpg"  # Test etmek istediğim görüntü
    
    if os.path.exists(test_image):
        test_model = YOLO(model_path) if os.path.exists(model_path) else YOLO(best_model_path)
        
        # Görüntü üzerinde tahmin yapma
        results = test_model.predict(test_image, save=True, conf=0.25)
        
        for r in results:
            print(f"Toplam {len(r.boxes)} adet nesne tespit edildi.")
            
            img = cv2.imread(test_image)
            
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            plt.figure(figsize=(10, 6))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title("Plaka Tespiti")
            plt.show()
            
            print("Test başarıyla tamamlandı ve sonuçlar gösterildi!")
    else:
        print(f"Test görüntüsü bulunamadı: {test_image}")

# Veri seti dışındaki herhangi bir resmi test etme
def test_new_image(image_path=None):
    model_path = "saved_models/plaka_tanima_model.pt"
    best_model_path = os.path.join(os.getcwd(), "runs/detect/train/weights/best.pt")
    
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Model '{model_path}' yüklendi.")
    elif os.path.exists(best_model_path):
        model = YOLO(best_model_path)
        print(f"Model '{best_model_path}' yüklendi.")
    elif os.path.exists("best.pt"):
        model = YOLO("best.pt")
        print("Model 'best.pt' yüklendi.")
    else:
        print("Eğitilmiş model bulunamadı! Önce modeli eğitmeniz gerekmektedir.")
        return
    
    if image_path is None:
        image_path = input("Lütfen test etmek istediğiniz görüntünün tam yolunu girin: ")
    
    if not os.path.exists(image_path):
        print(f"Hata: '{image_path}' konumunda görüntü bulunamadı!")
        return
    
    # tahmin
    print(f"'{image_path}' üzerinde tahmin yapılıyor...")
    results = model.predict(image_path, save=True, conf=0.25)
    
    # Sonuçlar
    for r in results:
        print(f"Toplam {len(r.boxes)} adet plaka tespit edildi.")
        
        if len(r.boxes) > 0:
            img = cv2.imread(image_path)
            
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Plaka
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            plt.figure(figsize=(10, 6))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title("Plaka Tespiti")
            plt.show()
            
            print("Test başarıyla tamamlandı ve sonuçlar gösterildi!")

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_img = img[y1:y2, x1:x2]

                plt.figure(figsize=(8, 4))
                plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                plt.imshow(plate_rgb)
                plt.axis('off')
                plt.title("Tespit Edilen Plaka")
                plt.show()
                

        else:
            print("Görüntüde plaka tespit edilemedi.")

if __name__ == '__main__':
    # Önceden çalışan tüm fonksiyonları kaldırdım ve sadece menüyü çalıştırıyorum
    
    print("Ne yapmak istediğinizi seçin:")
    print("1 - Modeli eğit (kısa süre)")
    print("2 - Modeli tam veri setiyle eğit (uzun süre)")
    print("3 - Modeli eğit, kaydet ve test et")
    print("4 - Yeni bir resimle test et")
    
    secim = input("Seçiminiz (1-4): ")
    
    if secim == "1":
        train()
    elif secim == "2":
        train_full_dataset()
    elif secim == "3":
        save_and_test_model()
    elif secim == "4":
        test_new_image()
    else:
        print("Geçersiz seçim!")

#GPU => Ekran kartı
#CPU => işlemci