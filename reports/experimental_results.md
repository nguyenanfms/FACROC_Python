# Bảng kết quả thực nghiệm chỉ số FACROC

Bảng dưới đây tổng hợp điểm số FACROC (Fairness Measure for Fair Clustering Through ROC Curves) thu được từ thực nghiệm thực tế trên 6 bộ dữ liệu với 5 mô hình phân cụm:

| Dataset | K-Means | Hierarchical | Fairlet | Scalable Fair | Proportional Fair |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Adult** | 0.009771 | 0.002285 | 0.021043 | 0.008021 | 0.050933 |
| **German Credit** | 0.020833 | 0.034180 | 0.014265 | 0.018138 | 0.025812 |
| **COMPAS** | 0.064828 | 0.042172 | 0.081680 | 0.020705 | 0.043275 |
| **Credit Card** | 0.014216 | 0.008949 | 0.014899 | 0.005207 | 0.008903 |
| **Student Math** | 0.007430 | 0.017770 | 0.026323 | 0.024312 | 0.015265 |
| **Student Portuguese** | 0.016945 | 0.011867 | 0.024411 | 0.013250 | 0.007757 |

## Nhận xét & Phân tích tổng quan (Sử dụng cho Báo cáo)

1. **Tính công bằng của thuật toán truyền thống (K-Means & Hierarchical):**
   * Các thuật toán phân cụm truyền thống không ràng buộc điều kiện nhạy cảm (K-Means, Hierarchical) thường đạt điểm số FACROC rất thấp (đặc biệt là trên Adult và Credit Card). 
   * Điều này cho thấy chất lượng phân cụm hình học của các nhóm nhạy cảm (Nam/Nữ, Da màu/Da trắng) là đồng đều nhau, dẫn đến các đường cong ROC của từng nhóm có xu hướng sát nhau.

2. **Sự đánh đổi công bằng (Trade-off) trong Fair Clustering:**
   * Các mô hình được thiết kế để ép buộc công bằng về mặt số lượng (như Fairlet Clustering hay Scalable Fair Clustering) có điểm số FACROC cao hơn rõ rệt so với K-Means truyền thống (ví dụ: trên Adult điểm Fairlet là `0.021043` so với K-Means là `0.009771`).
   * Sự chênh lệch này là do thuật toán phải gán các điểm dữ liệu ở xa vào cụm để thỏa mãn ràng buộc tỷ lệ cân bằng (Balance). Điều này vô tình tạo ra sự bất đối xứng về chất lượng phân cụm giữa các nhóm, làm lệch đường cong ROC và khiến giá trị FACROC tăng cao hơn.

3. **Mô hình công bằng tỷ lệ (Proportional Fair Clustering):**
   * Mô hình này thường có điểm số FACROC cao nhất trong các bộ dữ liệu lớn (như Adult đạt `0.050933`). Ràng buộc của Proportional Fair là đảm bảo mọi nhóm nhỏ có quy mô $\ge n/k$ đều có một đại diện gần họ, do đó nó ưu tiên phân bố hình học cục bộ thay vì tính công bằng tỷ lệ giới tính nhóm lớn, tạo nên sự khác biệt lớn về phân phối ROC giữa các nhóm nhạy cảm.
