# Multiple Linear Regression for Taxi Fare Prediction: An Empirical Study on NYC Yellow Taxi Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course: Statistical Inference](https://img.shields.io/badge/Course-Statistical%20Inference-green.svg)](https://sami.hust.edu.vn/)

## 1. Tổng quan đề tài (Abstract)

Dự án này trình bày nghiên cứu thực nghiệm về việc ứng dụng **Mô hình Hồi quy Tuyến tính Bội (Multiple Linear Regression - MLR)** và phương pháp ước lượng **Bình phương nhỏ nhất (Ordinary Least Squares - OLS)** nhằm phân tích và lượng hóa các nhân tố tác động đến giá cước cơ bản của dịch vụ taxi truyền thống - Yellow Taxi tại Thành phố New York.

Dựa trên dữ liệu thực tế do Ủy ban Taxi và Limousine Thành phố New York (NYC TLC) công bố, nghiên cứu thực hiện quy trình suy luận thống kê chuẩn mực bao gồm: tiền xử lý và lọc ngoại lai bằng khoảng tứ phân vị ($IQR$), phân tích tương quan Pearson, kiểm định giả thuyết thống kê Fisher ($F$-test) và Student ($t$-test), kiểm tra chẩn đoán các giả định Gauss-Markov trên phần dư, và xây dựng khoảng dự đoán với độ tin cậy $95\%$.

---

## 2. Thông tin tác giả & Khóa luận

* **Đề tài:** Mô hình hồi quy tuyến tính bội trong dự đoán giá cước dịch vụ taxi: Nghiên cứu thực nghiệm từ dữ liệu Yellow Taxi tại Thành phố New York.
* **Đơn vị:** Khoa Toán - Tin, Trường Đại học Bách khoa Hà Nội.
* **Sinh viên thực hiện:** Đào Tất Thành.

---

## 3. Cơ sở lý thuyết & Phương pháp luận (Theoretical Framework)

### 3.1. Mô hình tổng quát dạng ma trận

Xét mô hình hồi quy tuyến tính bội với biến phụ thuộc $Y \in \mathbb{R}^{n \times 1}$ và $k$ biến độc lập $X \in \mathbb{R}^{n \times p}$ (với $p = k + 1$):

$$Y = X\beta + \epsilon$$

Trong đó:
* $Y$: Vector quan sát của biến phụ thuộc (`fare_amount`).
* $X$: Ma trận thiết kế kích thước $n \times p$ với cột đầu tiên là vector đơn vị $\mathbf{1}$.
* $\beta = [\beta_0, \beta_1, \dots, \beta_k]^T$: Vector các hệ số hồi quy cần ước lượng.
* $\epsilon = [\epsilon_1, \epsilon_2, \dots, \epsilon_n]^T$: Vector sai số ngẫu nhiên thỏa mãn các giả định Gauss-Markov: $\mathbb{E}(\epsilon|X) = \mathbf{0}$ và $\text{Var}(\epsilon|X) = \sigma^2 I_n$.

### 3.2. Ước lượng OLS và suy luận thống kê

* **Ước lượng hệ số hồi quy:**
  $$\hat{\beta} = (X^T X)^{-1} X^T Y$$

* **Ước lượng không chệch của phương sai sai số $\sigma^2$:**
  $$s^2 = \frac{\text{SSE}}{n - p} = \frac{(Y - X\hat{\beta})^T (Y - X\hat{\beta})}{n - k - 1}$$

* **Kiểm định sự phù hợp toàn phần (F-test):**
  $$H_0: \beta_1 = \beta_2 = \dots = \beta_k = 0 \quad \text{vs} \quad H_1: \exists j \text{ sao cho } \beta_j \neq 0$$
  Thống kê kiểm định:
  $$F_0 = \frac{\text{MSR}}{\text{MSE}} = \frac{\text{SSR} / k}{\text{SSE} / (n - k - 1)} \sim F(k, n - p)$$

* **Kiểm định hệ số riêng phần (t-test):**
  $$H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0$$
  Thống kê kiểm định:
  $$T_0 = \frac{\hat{\beta}_j}{\text{se}(\hat{\beta}_j)} = \frac{\hat{\beta}_j}{s \sqrt{C_{jj}}} \sim t(n - p)$$
  với $C_{jj}$ là phần tử đường chéo thứ $j$ của ma trận $(X^T X)^{-1}$.

* **Khoảng dự đoán $100(1-\alpha)\%$ cho một quan sát mới $x_0$:**
  $$\hat{y}_0 \pm t_{\alpha/2, n-p} \cdot s \sqrt{1 + x_0^T (X^T X)^{-1} x_0}$$

---

## 4. Dữ liệu & Xử lý dữ liệu (Data Pipeline)

* **Nguồn dữ liệu:** NYC Taxi and Limousine Commission (TLC) Trip Record Data.
* **Cỡ mẫu:** $n_0 = 5000$ mẫu ngẫu nhiên $\to$ Sau khi lọc bỏ các điểm bất thường và ngoại lai bằng kỹ thuật Interquartile Range ($IQR = Q_3 - Q_1$, loại bỏ giá trị ngoài $[Q_1 - 1.5 IQR, Q_3 + 1.5 IQR]$), tập mẫu chính thức gồm $n = 4541$ quan sát.

### Không gian biến số

| Ký hiệu | Tên biến | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- | :--- |
| $Y$ | `gia_cuoc_cb` | Liên tục (USD) | Giá cước cơ bản tính theo công tơ mét (`fare_amount`) |
| $X_1$ | `do_dai_qd` | Liên tục (dặm) | Quãng đường di chuyển ghi nhận qua GPS (`trip_distance`) |
| $X_2$ | `thoi_gian_ht` | Liên tục (phút) | Thời gian thực hiện hành trình (`duration_min`) |
| $X_3$ | `gio_cao_diem` | Nhị phân ($\{0, 1\}$) | $1$ nếu chuyến đi trong khung 7-9h hoặc 16-19h; $0$ nếu khác |
| $X_4$ | `cuoi_tuan` | Nhị phân ($\{0, 1\}$) | $1$ nếu chuyến đi vào Thứ Bảy / Chủ Nhật; $0$ nếu ngày thường |

---

## 5. Kết quả thực nghiệm & Chẩn đoán mô hình

### 5.1. Phân tích tương quan & Loại bỏ biến

* Tương quan tuyến tính Pearson giữa `cuoi_tuan` và `gia_cuoc_cb` đạt $r \approx 0.00$, phản ánh tính độc lập tuyến tính giữa cước phí cơ bản và ngày trong tuần $\implies$ loại khỏi mô hình ban đầu.
* Tại mô hình đầy đủ, biến `gio_cao_diem` cho giá trị kiểm định $t_0 = -0.8258$ ($|t_0| < t_{0.025; 4536} = 1.9605$, $p\text{-value} > 0.05$). Do đó, ta không bác bỏ giả thuyết $H_0$, xác nhận phụ phí giờ cao điểm được hệ thống TLC tách rời khỏi giá cước nền hoặc đã phản ánh gián tiếp qua thời gian hành trình $\implies$ loại bỏ bằng thuật toán lùi (Backward Elimination).

### 5.2. Mô hình hồi quy tối ưu

Mô hình cuối cùng với hai biến giải thích $X_1$ (`do_dai_qd`) và $X_2$ (`thoi_gian_ht`):

$$\hat{Y} = 2.2384 + 2.3588 \cdot X_1 + 0.5566 \cdot X_2$$

| Biến số / Tham số | Hệ số ($\hat{\beta}$) | Sai số chuẩn ($\text{se}$) | $t$-statistic | $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **Hệ số chặn** ($\beta_0$) | $2.2384$ | $0.027$ | $82.90$ | $< 10^{-16}$ |
| **Quãng đường** ($X_1$) | $2.3588$ | $0.010$ | $235.88$ | $< 10^{-16}$ |
| **Thời gian** ($X_2$) | $0.5566$ | $0.002$ | $278.30$ | $< 10^{-16}$ |

* **Độ phù hợp của mô hình:**
  * Hệ số xác định: $R^2 = 0.986$
  * Hệ số xác định hiệu chỉnh: $R^2_{\text{adj}} = 0.986$
  * Kiểm định Fisher toàn phần: $F_0 = 82396.08 > F_{0.05}(4, 4536) = 2.3739$ ($p < 10^{-16}$) $\implies$ Bác bỏ $H_0$.

### 5.3. Chẩn đoán phần dư (Residual Diagnostics)

* **Tính chuẩn của sai số:** Biểu đồ Histogram và Q-Q Plot cho thấy phần dư tập trung đối xứng quanh điểm 0. Mặc dù có hiện tượng đuôi nặng nhẹ (leptokurtic) do các chuyến đi đặc thù, định lý giới hạn trung tâm (CLT) với cỡ mẫu $n = 4541$ vẫn đảm bảo tính tiệm cận chuẩn của các ước lượng.
* **Tính đồng nhất phương sai (Homoscedasticity):** Biểu đồ Fitted Values vs Residuals thể hiện sự phân tán đều quanh trục sai số bằng 0, không có dạng hình phễu hay phân kì mang tính hệ thống.

---



---

