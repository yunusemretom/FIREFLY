import os
from rfdetr import RFDETRSmall


def main() -> None:
    model = RFDETRSmall(
        pretrain_weights="/home/tom/Downloads/checkpoint_best_regular (2).pth",
        num_classes=3,
        num_queries=300,
    )
    output_dir = "/home/tom/Documents/FIREFLY/object_detection"
    exported_path = model.export(
        output_dir=output_dir,
    )
    
    target_path = os.path.join(output_dir, "inference_model.onnx")
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(exported_path, target_path)
    print(f"ONNX export tamamlandi: {target_path}")


if __name__ == "__main__":
    main()