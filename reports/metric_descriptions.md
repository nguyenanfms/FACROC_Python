# Hướng Dẫn Tính Toán Các Chỉ Số: Đường Cong ROC, AUCC và FACROC trong Phân Cụm

Tài liệu này mô tả chi tiết cơ sở toán học và quy trình tính toán các chỉ số đánh giá chất lượng và tính công bằng trong phân cụm phi giám sát được áp dụng trong đồ án, bao gồm: **Đường cong ROC trong phân cụm**, chỉ số **AUCC (Area Under the Curve for Clustering)**, và độ đo công bằng **FACROC**.

---

## 1. Đường Cong ROC trong Phân Cụm (Clustering ROC Curve)

### Ý tưởng cốt lõi
Trong bài toán phân loại có giám sát, đường cong ROC (Receiver Operating Characteristic) biểu diễn mối quan hệ giữa **True Positive Rate (TPR)** và **False Positive Rate (FPR)** tương ứng với từng ngưỡng xác suất phân loại.

Tuy nhiên, đối với bài toán phân cụm phi giám sát (unsupervised clustering), chúng ta **không có nhãn lớp thực tế (ground truth labels)** để tính toán TPR và FPR theo cách truyền thống. Để giải quyết vấn đề này, chất lượng phân cụm được đánh giá gián tiếp thông qua **quan hệ cặp (pairwise relationships)**:
*   Mỗi cặp điểm dữ liệu $(x_i, x_j)$ được xem là một đối tượng đánh giá.
*   **Điểm tương đồng (Similarity Score)** giữa hai điểm được xác định dựa trên khoảng cách không gian gốc.
*   **Nhãn dự đoán nhị phân (Binary Prediction)** thể hiện việc thuật toán có xếp hai điểm này vào chung một cụm hay không.

---

## 2. Chỉ số AUCC (Area Under the Curve for Clustering)

AUCC là diện tích dưới đường cong ROC được thiết lập cho bài toán phân cụm. Nó đo lường mức độ đồng nhất giữa **khoảng cách hình học thực tế** và **kết quả gán cụm** của thuật toán. Một AUCC cao (gần 1.0) cho thấy thuật toán đã gom các điểm ở gần nhau vào chung một cụm một cách nhất quán.

### Cơ sở Toán học & Các bước tính toán

Giả sử tập dữ liệu đầu vào là $\mathcal{X} = \{x_1, x_2, \dots, x_n\}$ và kết quả phân cụm thu được là các nhãn cụm $\mathcal{C} = \{c_1, c_2, \dots, c_n\}$.

#### Bước 1: Tính ma trận khoảng cách cặp (Pairwise Distance)
Tính khoảng cách Euclidean giữa mọi cặp điểm dữ liệu $(x_i, x_j)$:
$$d_{ij} = \|x_i - x_j\|_2$$

#### Bước 2: Chuyển đổi thành không gian cặp (Pairwise Space)
Xét tất cả các cặp điểm duy nhất $(x_i, x_j)$ với $i < j$ (nửa tam giác trên của ma trận khoảng cách, gồm $N_{pairs} = \frac{n(n-1)}{2}$ cặp):
1.  **Mảng độ tương đồng (Similarity Score - $S_{ij}$):** Khoảng cách càng nhỏ thì độ tương đồng càng cao. Do đó, ta đặt điểm tương đồng là giá trị âm của khoảng cách:
    $$S_{ij} = -d_{ij}$$
2.  **Mảng nhãn phân cụm nhị phân (Binary Clustering Label - $Y_{ij}$):** Đóng vai trò là nhãn thực tế (Ground Truth) phục vụ vẽ ROC:
    $$Y_{ij} = \begin{cases} 1 & \text{nếu } c_i = c_j \text{ (chung cụm)} \\ 0 & \text{nếu } c_i \neq c_j \text{ (khác cụm)} \end{cases}$$

#### Bước 3: Sắp xếp và xác định các ngưỡng (Thresholding)
Sắp xếp tất cả các cặp theo thứ tự độ tương đồng giảm dần (tương đương với khoảng cách tăng dần):
$$S^{(1)} \geq S^{(2)} \geq \dots \geq S^{(N_{pairs})}$$
Tương ứng, mảng nhãn phân cụm cũng được sắp xếp lại thành $Y^{(1)}, Y^{(2)}, \dots, Y^{(N_{pairs})}$.

#### Bước 4: Tính toán TPR và FPR tại mỗi ngưỡng $t$
Với mỗi ngưỡng độ tương đồng $t$, các cặp có $S_{ij} \geq t$ được dự đoán là "chung cụm" (Positive), các cặp có $S_{ij} < t$ được dự đoán là "khác cụm" (Negative).
*   **True Positives (TP):** Số cặp thực sự chung cụm và được xếp chung cụm.
*   **False Positives (FP):** Số cặp thực tế khác cụm nhưng bị xếp chung cụm.

Từ đó, ta tính toán tỷ lệ tại mỗi ngưỡng:
$$TPR(t) = \frac{TP(t)}{\text{Tổng số cặp chung cụm (Positive)}}$$
$$FPR(t) = \frac{FP(t)}{\text{Tổng số cặp khác cụm (Negative)}}$$

#### Bước 5: Tính tích phân AUCC
Đường cong ROC được vẽ bằng cách nối các điểm tọa độ $(FPR(t), TPR(t))$. Giá trị AUCC được tính bằng tích phân diện tích dưới đường cong này bằng quy tắc hình thang (Trapezoidal Rule):
$$AUCC = \sum_{k=1}^{M} \frac{TPR^{(k)} + TPR^{(k-1)}}{2} \times \left(FPR^{(k)} - FPR^{(k-1)}\right)$$

---

## 3. Độ Đo Công Bằng FACROC

FACROC (Fairness Area under the Curve for ROC of Clustering) đo lường độ lệch công bằng về chất lượng phân cụm giữa nhóm nhạy cảm được bảo vệ (Protected Group - $P$) và nhóm không được bảo vệ (Non-Protected Group - $U$).

### Quy trình tính toán

#### Bước 1: Tính toán đường cong ROC riêng cho từng nhóm
Tách dữ liệu thành hai phần dựa trên thuộc tính nhạy cảm:
1.  **Nhóm bảo vệ ($P$):** Tính toán và thu được các cặp tọa độ đường cong $ROC_P = (FPR_P, TPR_P)$.
2.  **Nhóm không bảo vệ ($U$):** Tính toán và thu được các cặp tọa độ đường cong $ROC_U = (FPR_U, TPR_U)$.

#### Bước 2: Nội suy tuyến tính (Interpolation) lên lưới chung
Vì số lượng phần tử của hai nhóm khác nhau dẫn đến số điểm tọa độ trên hai đường cong ROC khác nhau, ta thực hiện nội suy tuyến tính (Linear Interpolation) cả hai đường cong lên một lưới tọa độ FPR chung gồm $M$ điểm (ví dụ $M = 40,000$ điểm trải đều từ $0.0$ đến $1.0$):
*   Nội suy thu được $ROC_P(t)$ tại mỗi điểm $t \in [0, 1]$.
*   Nội suy thu được $ROC_U(t)$ tại mỗi điểm $t \in [0, 1]$.

#### Bước 3: Tính toán FACROC
FACROC chính là phần diện tích nằm xen giữa hai đường cong ROC của hai nhóm. Công thức tích phân xác định FACROC là:
$$FACROC = \int_{0}^{1} | ROC_U(t) - ROC_P(t) | dt$$

Giá trị này được tính toán thực tế bằng cách lấy trung bình sai lệch tuyệt đối trên lưới điểm nội suy:
$$FACROC \approx \frac{1}{M} \sum_{k=1}^{M} | ROC_U(t_k) - ROC_P(t_k) |$$

### Ý nghĩa chỉ số:
*   $FACROC \in [0, 1]$.
*   **$FACROC \to 0$:** Chất lượng phân cụm hình học của hai nhóm hoàn toàn tương đồng $\rightarrow$ **Mô hình đạt tính công bằng tối đa**.
*   **$FACROC \to 1$:** Sự chênh lệch đối xử cực kỳ lớn giữa hai nhóm $\rightarrow$ **Mô hình mất công bằng nghiêm trọng**.
