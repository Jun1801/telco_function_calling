- Lên kịch bản các synthetic datasets: types gì
- với câu queries multi-scenerio, multi-intents:
    + Xử lý song song: So sánh số trạm 5G Hà Nội vs HCM -> trả về cùng hàm khác tham số
    + Tuần tự: Vị tổng thống thứ 15 của Mỹ sinh năm? -> đi tuần tự -> trả về tong_thong(15, Mỹ) + sinh_nam(tong_thong(15, Mỹ))
- Thu hẹp phạm vi giá trị của các tham số trong function call ->
- Tính lại so sánh các baselines
- taur dataset, ToolRL: Reward is All Tool Learning Need, tìm kiếm thêm các hàm ngoài
- https://llm-stats.com/benchmarks/tau-bench

- test: 1000-1500 samples
- 
- Ablation hơn (reward)
- Xây pipeline tự động hoá từ đấu đến cuối (tư training, tự sinh dữ liêu)
- Bước sinh dữ liệu cần làm sao để đảm bảo validations tốt nhất
- Retrieval