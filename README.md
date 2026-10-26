# Theo dõi biểu cảm thời gian thực

Ứng dụng desktop bật webcam, dò khuôn mặt và nhận diện 7 biểu cảm (angry, disgust, fear, happy, sad,
surprise, neutral) bằng model ONNX (ResNet50, train trên FER+/RAF-DB/AffectNet).

## Cài đặt (Windows, CMD)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Copy model vào thư mục `models\` (giữ nguyên tên file). Nếu model có kèm file `.onnx.data`
thì phải copy cả hai, hoặc gộp thành một file (xem phần "Gộp model" bên dưới).

## Chạy

    python main.py
    python main.py --model models\ten_model.onnx --cam 0 --every 3

| Tham số   | Ý nghĩa                                              | Mặc định |
|-----------|------------------------------------------------------|----------|
| `--model` | đường dẫn `.onnx` (bỏ trống: tự tìm trong `models\`)  | tự tìm   |
| `--cam`   | số thứ tự camera                                     | 0        |
| `--every` | cứ N khung hình đánh giá 1 lần                       | 3        |
| `--alpha` | hệ số làm mượt EMA (nhỏ = mượt hơn nhưng trễ hơn)    | 0.35     |

Mọi tham số khác (ngưỡng dò mặt, độ rộng viền, thời gian chờ mất mặt...) chỉnh trong
`emotion_tracker/config.py`.

## Cấu trúc

    main.py                     điểm vào, nối các thành phần
    emotion_tracker/
        config.py               cấu hình tập trung (dataclass) + đọc tham số dòng lệnh
        labels.py               nhãn, tên tiếng Việt, màu
        detector.py             dò khuôn mặt (YuNet, dự phòng Haar) + cắt mặt
        classifier.py           nạp ONNX, dự đoán, báo lỗi model rõ ràng
        worker.py               luồng suy luận riêng, không làm đứng giao diện
        smoothing.py            EMA + chống nhảy nhãn (thuần logic)
        pipeline.py             xử lý 1 khung hình: dò -> cắt -> đánh giá mỗi N khung -> làm mượt
        camera.py               đọc webcam, cắt 4:3, lật gương
        session.py              ghi CSV và tính tỉ lệ thời gian theo biểu cảm
        metrics.py              đo FPS / số lần đánh giá mỗi giây
        ui/panels.py            các panel vẽ: video, thanh xác suất, đồ thị, điều khiển
        ui/app.py               cửa sổ chính
    tests/                      pytest cho các module lõi

Quy tắc: chỉ `emotion_tracker/ui/` được import tkinter; các module còn lại test được không cần màn hình.

## Test

    pip install -r requirements-dev.txt
    python -m pytest

## Gộp model thành một file (khi có `.onnx.data`)

    import onnx
    m = onnx.load("ten_model.onnx")   # tự đọc kèm file .data cạnh nó
    onnx.save_model(m, "ten_model_single.onnx", save_as_external_data=False)
"# EmotionReg" 
