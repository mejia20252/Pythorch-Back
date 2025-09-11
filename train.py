from ultralytics import YOLO
import os
import glob

def verify_dataset(base):
    ok = True
    for split in ["train", "val"]:
        imgs = sorted(glob.glob(os.path.join(base, "images", split, "*.*")))
        missing = []
        for p in imgs:
            name = os.path.splitext(os.path.basename(p))[0] + ".txt"
            lbl  = os.path.join(base, "labels", split, name)
            if not os.path.exists(lbl):
                missing.append((p, lbl))
        print(f"[{split}] imgs={len(imgs)}  sin_label={len(missing)}")
        for i,(img,lbl) in enumerate(missing[:10], 1):
            print(f"  {i}. Falta label: {lbl}  (para {img})")
        if missing:
            ok = False
    return ok

def train():
    # Rutas
    repo_dir  = os.getcwd()  # C:\Users\Usuario\Documents\si2_projects\model-trainer-tensorflow
    data_dir  = os.path.join(repo_dir, "data")
    data_yaml = os.path.join(data_dir, "data.yaml")

    print("Usando data.yaml:", data_yaml)
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"No existe {data_yaml}")

    # Verifica que cada imagen tenga su .txt espejo
    if not verify_dataset(data_dir):
        print("⚠️ Corrige los labels faltantes antes de entrenar.")
        return

    # Modelo (v11 o v8 ambos sirven)
    model = YOLO("yolo11n.pt")  # o "yolov8n.pt"

    try:
        results = model.train(
            data=data_yaml,
            epochs=50,       # con 4/1 samples, menos épocas basta para probar
            imgsz=640,
            batch=4,
            exist_ok=True, 
            workers=0,       # Windows: evita problemas de DataLoader
            device="cpu",    # <-- CPU porque tu Torch no ve CUDA
            project="runs",
            name="plates_yolo11n"
        )
        print("✅ Training complete. Results object:", results)
    except Exception as e:
        # Para ver el error real sin romper por NameError
        print("❌ Error durante el entrenamiento:", repr(e))

if __name__ == "__main__":
    train()
