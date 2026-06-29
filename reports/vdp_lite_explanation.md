# Chi Tiết Về Thuật Toán VDP-lite (Variational Policy Distillation - Lite)

Tài liệu này giải thích chi tiết cơ chế hoạt động, công thức toán học và cách cài đặt của thuật toán **VDP-lite** phục vụ tối ưu hóa mô hình Telco Function Calling trong dự án.

---

## 1. Bản chất của VDP-lite

**VDP-lite** là một phương pháp tối ưu hóa chính sách RL dựa trên cơ chế chưng cất tri thức (Knowledge Distillation) bất đối xứng thông qua vòng lặp **Expectation-Maximization (EM)**. 

Kiến trúc gồm hai LoRA adapter chạy song song trên cùng một Base Model:
*   **Teacher Model ($\pi_\theta$)**: Được tiếp cận thông tin **ưu tiên (privileged context)** gồm `Prompt + Environment Feedback` (ví dụ: mô tả chi tiết lỗi cú pháp, sai tham số từ môi trường). Nhờ có feedback giải thích lý do sai, Teacher cực kỳ dễ dàng tìm ra hành động đúng.
*   **Student Model ($\pi_\phi$)**: Chỉ tiếp cận **standard context** gồm `Prompt` (không có feedback). Student phải tự học cách giải quyết tác vụ bằng cách bắt chước phân phối xác suất của Teacher.

---

## 2. Vòng lặp EM (Expectation - Maximization)

Quá trình huấn luyện luân phiên cập nhật giữa **E-step** và **M-step**:

### 2.1. E-step (Expectation Step): Tối ưu hóa Teacher ($\pi_\theta$)

Mục tiêu của E-step là cập nhật Teacher adapter sao cho tối đa hóa phần thưởng nhận được từ môi trường (rollouts), đồng thời giữ chính sách của Teacher không đi quá xa Student để đảm bảo tính chưng cất được.

*   **Hàm Loss ở E-step**:
    $$\mathcal{L}_E(\theta) = \mathcal{L}_{\text{Reward}}(\theta) + \beta \cdot D_{\text{KL}}(\pi_\theta \,\|\, \pi_\phi)$$
    *(Với $\beta$ đại diện cho hệ số `teacher_trust_region_beta`)*

*   **Reward Loss ($\mathcal{L}_{\text{Reward}}$)**:
    Tùy thuộc vào kết quả phần thưởng (Reward) của quỹ đạo $y$:
    *   **Nếu Rollout thành công** (Reward $\ge$ success\_threshold):
        $$\mathcal{L}_{\text{Reward}} = \lambda \cdot \text{Reward} \cdot \mathcal{L}_{\text{NLL}}(\theta)$$
        *(Học tăng cường quỹ đạo đúng, với $\lambda$ là trọng số progressive tăng dần theo thời gian huấn luyện).*
    *   **Nếu Rollout thất bại** (Reward < success\_threshold):
        $$\mathcal{L}_{\text{Reward}} = - w_{\text{neg}} \cdot \mathcal{L}_{\text{NLL}}(\theta)$$
        *(Phạt quỹ đạo sai - đẩy xác suất sinh chuỗi lỗi của Teacher ra xa).*

*   **KL Trust-Region Penalty**:
    Tính toán sai lệch KL $D_{\text{KL}}(\pi_\theta \,\|\, \pi_\phi)$ trên `top_k` token của Teacher (hàm `compute_top_k_kl` trong `train_vpd_hf.py`). Việc này ép phân phối của Teacher bám sát Student hiện tại, tránh hiện tượng Teacher tự tối ưu hóa ra một phân phối ngôn ngữ quá phức tạp khiến Student không thể học nổi.

### 2.2. M-step (Maximization Step): Tối ưu hóa Student ($\pi_\phi$)

Trong M-step, Teacher model được đóng băng. Student adapter sẽ học bắt chước (chưng cất) phân phối xác suất của Teacher từ E-step.

*   **Hàm Loss ở M-step**:
    $$\mathcal{L}_M(\phi) = \mathcal{L}_{\text{Distill}}(\phi) + w_{\text{anchor}} \cdot \mathcal{L}_{\text{Anchor}}(\phi)$$

*   **Distillation Loss (Jensen-Shannon Divergence - JSD)**:
    Thay vì dùng KL một chiều, VDP-lite sử dụng JSD giữa logits của Student và top-k logits của Teacher để đảm bảo độ mịn và ổn định của gradient:
    $$\text{JSD}(\pi_\phi, \pi_\theta) = (1 - \alpha) \cdot D_{\text{KL}}(\pi_\phi \,\|\, M) + \alpha \cdot D_{\text{KL}}(\pi_\theta \,\|\, M)$$
    *(Với $M$ là phân phối trộn: $M = (1-\alpha)\pi_\phi + \alpha\pi_\theta$)*

*   **Importance Sampling (IS) Correction**:
    Vì các rollout được tạo ra bởi một phiên bản chính sách cũ (off-policy), hệ thống nhân thêm trọng số IS để điều chỉnh gradient về đúng phân phối hiện tại:
    $$w_{\text{IS}} = \exp(\log \pi_\phi(y) - \log \pi_{\text{rollout}}(y))$$
    Trọng số này được kẹp tối đa là `is_clip` (thường là 2.0) để tránh bùng nổ gradient. Loss chưng cất thực tế sẽ là:
    $$\mathcal{L}_{\text{Distill}} = w_{\text{IS}} \cdot \text{JSD}(\pi_\phi, \pi_\theta)$$

*   **Anchor Loss (Regularization)**:
    Để chống lại hiện tượng quên thảm họa (catastrophic forgetting) các tác vụ cơ bản (như định dạng JSON hoặc trường hợp từ chối - abstain), một mẫu ngẫu nhiên từ tập SFT gốc (`anchor_file`) được đưa vào tính NLL loss truyền thống và cộng gộp vào gradient của Student với trọng số `anchor_weight` (thường là 0.2).

---

## 3. Tóm tắt luồng hoạt động của Code

Quy trình huấn luyện luân phiên trong file `scripts/train_vpd_hf.py` diễn ra như sau:

| Giai đoạn | Hàm thực thi | Trạng thái mô hình | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| **E-Step** | `e_step()` | Teacher `train` / Student `eval` | Cập nhật **Teacher adapter** dựa trên Reward của rollout + KL trust-region phạt khi Teacher lệch quá xa Student. |
| **M-Step** | `m_step()` | Student `train` / Teacher `eval` | Cập nhật **Student adapter** qua JSD distillation bắt chước Teacher, áp dụng Importance Sampling và Anchor Loss. |
