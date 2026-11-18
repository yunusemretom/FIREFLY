# object_detection/

RF-DETR tabanlı Talon dedektörünün model dosyası burada durur.
Dosya büyük olduğu için (115 MB, GitHub'ın 100 MB sınırının üstünde) **repoya girmez**
(`.gitignore`). Ekip içinde ayrı paylaşılır.

| Dosya | Açıklama |
|---|---|
| `inference_model.onnx` | RF-DETR, 512×512 giriş. Çıkış: `dets [1,300,4]` (cx,cy,w,h normalize), `labels [1,300,3]` logit. Simülatörde Talon = sınıf 1. |

Kullanan kod: `src/python/vision/talon_detector.py` (varsayılan yol bu klasör).
GPU ile çalıştırma için bkz. [`docs/simulator_ve_ortam.md`](../docs/simulator_ve_ortam.md).
