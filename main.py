# YoloV8
from ultralytics import YOLO
# yolov8n => bir computer vision modeli
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics.utils.callbacks.tensorboard import on_fit_epoch_end
from ultralytics.utils.callbacks import Callbacks

def train():
    model = YOLO("yolov8n.pt")
    
    # Checkpoint callback tanımlama
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks oluşturma
    callbacks = Callbacks(model)
    
    # ModelCheckpoint callback ekleme
    def save_checkpoint(trainer):
        epoch = trainer.epoch
        model_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        trainer.model.save(model_path)
        print(f"Checkpoint kaydedildi: {model_path}")
    
    # Callback'i kaydet
    callbacks.register_action("on_train_epoch_end", save_checkpoint)
    
    # Daha hızlı eğitim için epochs değerini 3'e düşürdüm
    model.train(data="data.yaml", epochs=3, imgsz=320, batch=4, patience=5, 
                callbacks=callbacks)

# Pythonda starndart koruma yapısı.
# python {main.py} => bu kodu çalıştırır.
# ama diğer dosyalar import ettiğinde burayı çalıştırmaz.
#if __name__ == '__main__':
#    train()

####
#full dataset ile train
def train_full_dataset():
    model = YOLO("yolov8n.pt")
    
    # Checkpoint callback tanımlama
    checkpoint_dir = "checkpoints_full"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks oluşturma
    callbacks = Callbacks(model)
    
    # ModelCheckpoint callback ekleme
    def save_checkpoint(trainer):
        epoch = trainer.epoch
        model_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        trainer.model.save(model_path)
        print(f"Checkpoint kaydedildi: {model_path}")
    
    # Callback'i kaydet
    callbacks.register_action("on_train_epoch_end", save_checkpoint)
    
    # epochs:10, imgsz (görüntü boyutu): 320x320, batch:16
    results = model.train(data="data.yaml", epochs=10, imgsz=320, batch=16, patience=5,
                         callbacks=callbacks)
    
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
    model_paths = [
        "saved_models/plaka_tanima_model.pt",
        os.path.join(os.getcwd(), "runs/detect/train/weights/best.pt"),
        os.path.join(os.getcwd(), "runs/detect/train7/weights/best.pt"),
        "best.pt"
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = YOLO(path)
            print(f"Model '{path}' yüklendi.")
            break
    
    if model is None:
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

def resume_training_from_checkpoint(checkpoint_path=None, epochs=5):
    """
    Checkpoint'ten eğitime devam etmek için fonksiyon
    
    Args:
        checkpoint_path: Checkpoint dosyasının yolu
        epochs: Devam edilecek epoch sayısı
    """
    # Checkpoint yolu belirtilmemişse en son checkpoint'i bul
    if checkpoint_path is None:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoint_files:
                # En son epoch'u bul
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
                print(f"En son checkpoint bulundu: {checkpoint_path}")
            else:
                print("Checkpoint bulunamadı, lütfen checkpoint yolunu belirtin.")
                return
        else:
            print("Checkpoint dizini bulunamadı, lütfen checkpoint yolunu belirtin.")
            return
    
    # Checkpoint'ten modeli yükle
    if not os.path.exists(checkpoint_path):
        print(f"Hata: '{checkpoint_path}' bulunamadı!")
        return
    
    model = YOLO(checkpoint_path)
    print(f"Model '{checkpoint_path}' yüklendi.")
    
    # Checkpoint callback tanımlama
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Callbacks oluşturma
    callbacks = Callbacks(model)
    
    # ModelCheckpoint callback ekleme
    def save_checkpoint(trainer):
        epoch = trainer.epoch
        model_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        trainer.model.save(model_path)
        print(f"Checkpoint kaydedildi: {model_path}")
    
    # Callback'i kaydet
    callbacks.register_action("on_train_epoch_end", save_checkpoint)
    
    # Eğitime devam et
    model.train(data="data.yaml", epochs=epochs, imgsz=320, batch=4, patience=5, 
                callbacks=callbacks)
    
    print(f"Eğitim tamamlandı! Model otomatik olarak runs/detect/train klasörüne kaydedildi.")

if __name__ == '__main__':
    # Önceden çalışan tüm fonksiyonları kaldırdım ve sadece menüyü çalıştırıyorum
    
    print("Ne yapmak istediğinizi seçin:")
    print("1 - Modeli eğit (kısa süre)")
    print("2 - Modeli tam veri setiyle eğit (uzun süre)")
    print("3 - Modeli eğit, kaydet ve test et")
    print("4 - Yeni bir resimle test et")
    print("5 - Checkpoint'ten eğitime devam et")
    
    secim = input("Seçiminiz (1-5): ")
    
    if secim == "1":
        train()
    elif secim == "2":
        train_full_dataset()
    elif secim == "3":
        save_and_test_model()
    elif secim == "4":
        test_new_image()
    elif secim == "5":
        checkpoint_path = input("Lütfen devam etmek istediğiniz checkpoint dosyasının yolunu girin: ")
        resume_training_from_checkpoint(checkpoint_path)
    else:
        print("Geçersiz seçim!")

#GPU => Ekran kartı
#CPU => işlemci