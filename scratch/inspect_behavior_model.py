from ultralytics import YOLO
import sys

def main():
    model_path = "models/behavior.pt"
    try:
        model = YOLO(model_path)
        print("=== Behavior Model Inspection ===")
        print("Task Type:", model.task)
        print("Class Names (model.names):")
        for k, v in model.names.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
