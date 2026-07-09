from pathlib import Path

from rfdetr import RFDETRSmall

OUTPUT_DIR  = Path("/home/tom/Documents/FIREFLY/object_detection")
OUTPUT_NAME = "inference_model.onnx"


def main() -> None:
    model = RFDETRSmall(pretrain_weights="/home/tom/Downloads/checkpoint_best_regular (09.08.16).pth")

    # export() sadece output_dir alir; dosyayi model.onnx olarak kaydeder
    result_path: Path = model.export(output_dir=str(OUTPUT_DIR))

    # istenen isimle yeniden adlandir
    target = OUTPUT_DIR / OUTPUT_NAME
    if result_path != target:
        result_path.rename(target)

    print(f"ONNX export tamamlandi: {target}")


if __name__ == "__main__":
    main()