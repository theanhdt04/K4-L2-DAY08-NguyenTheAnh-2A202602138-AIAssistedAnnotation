# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Thế Anh

Công cụ gán nhãn đã dùng: CVAT Docker local

Mọi con số trong báo cáo phải truy được từ `reports/rounds_table.md`,
`outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Video là một cảnh quay liên tục với camera cố định nên các frame gần nhau chứa cùng chiếc xe trong nhiều thời điểm. Pool và test được chia theo trục thời gian và có vùng đệm để hạn chế việc cùng một xe xuất hiện ở cả hai tập. Nếu chia ngẫu nhiên, dữ liệu sẽ bị rò rỉ: mô hình có thể gặp gần như cùng cảnh hoặc cùng xe khi train rồi được chấm lại trên test, làm AP50, precision và recall cao hơn khả năng tổng quát thực tế. Cách chia theo thời gian khó hơn nhưng phản ánh tốt hơn việc dự đoán trên thời điểm chưa thấy.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

Vòng 0: `yolov8n cold start (COCO car+bus+truck)`, 0 ảnh train, 0 box train, AP50 `0.771`, P `0.925`, R `0.489`, F1 `0.640`, R small `0.182`, R medium `0.547`, R large `0.561`.

Ảnh `compare_round0.jpg` cho thấy mô hình bắt được nhiều xe có thân và đèn rõ ở gần camera nhưng bỏ sót hoặc đặt box chưa khớp với các xe nhỏ ở xa, xe tối và xe chỉ còn cụm đèn/taillight. Recall small chỉ `0.1818` trên 66 box, thấp hơn nhiều so với medium `0.5473` trên 296 box và large `0.5610` trên 41 box. Một trường hợp cần người rà nhãn tham chiếu trước khi kết luận mô hình sai là các điểm sáng rất nhỏ ở đường chân trời: chúng có thể là xe thật nhưng cũng có thể là phản chiếu đèn hoặc vật sáng; hơn nữa 14 box cao dưới 16 pixel được quy tắc đánh giá bỏ qua.

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công thức `score = W_U·U + W_A·A + W_D·D` kết hợp ba tín hiệu: `U` là độ bất định của năm box khó nhất, `A` là tỷ lệ box có confidence mập mờ, còn `D` là độ đa dạng theo thời gian so với ảnh đã chọn. Trong `tools/al_select.py`, trọng số mặc định là `W_U = 0.5`, `W_A = 0.3`, `W_D = 0.2`. `MIN_GAP_S = 2.0` tránh các frame cách frame đã chọn dưới 2 giây vì camera cố định khiến chúng gần như trùng cảnh và tốn công rà nhưng ít thông tin mới.

Ba frame trong `SELECTION.md` minh họa các ưu tiên khác nhau: `frame_0182.jpg` đứng hạng 1 với score `0.9591`, 18 box mơ hồ trong 28 box; `frame_0369.jpg` có U `0.9315` và 16 box mơ hồ trong 43 box; `frame_0099.jpg` có U `0.9460` và 14 box mơ hồ trong 29 box. `frame_0331.jpg` có score cao `0.9154` nhưng cách `frame_0326.jpg` đúng 2.0 giây, nên tôi không ưu tiên cả hai nếu chỉ có ngân sách rà năm ảnh. Điểm bất định chỉ cho biết nơi model đang phân vân để ưu tiên rà nhãn, không chứng minh ảnh đó chắc chắn cải thiện mô hình; hiệu quả còn phụ thuộc chất lượng box sau khi người rà sửa.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

`rounds_table.md` có hai vòng:

| vòng | model                                   | ảnh train | box train |  AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 |    F1 | R small | R medium | R large |
| ---: | --------------------------------------- | --------: | --------: | ----: | -------------------: | -----: | -----: | ----: | ------: | -------: | ------: |
|    0 | yolov8n cold start (COCO car+bus+truck) |         0 |         0 | 0.771 |                    — |  0.925 |  0.489 | 0.640 |   0.182 |    0.547 |   0.561 |
|    1 | yolov8n fine-tune vong 1..1             |        12 |       357 | 0.475 |               -0.297 |  1.000 |  0.045 | 0.086 |   0.000 |    0.037 |   0.171 |

Theo `outputs/round1_diff.md`, vòng 1 dùng 12 ảnh; model đề xuất 169 box và nhãn sau rà có 357 box. Có 128 box được giữ gần như nguyên, 25 box được chỉnh, 16 box bị xóa và 204 box được thêm mới; accept rate là 76%. Các con số này cho thấy pre-label bỏ sót nhiều xe trong cảnh đêm đông xe. `REVIEW_LOG.csv` ghi ba ca cụ thể: thêm xe bị cắt ở góc dưới phải trong `frame_0099.jpg`, xóa vùng sáng không phải xe trong `frame_0182.jpg`, và chỉnh box xe tối gần mép phải trong `frame_0187.jpg`.

Theo `outputs/metrics_round0.json` và `outputs/metrics_round1.json`, model vòng 1 giảm AP50 từ `0.771` xuống `0.475`, tức Δ AP50 `-0.297`; recall giảm từ `0.489` xuống `0.045`, F1 từ `0.640` xuống `0.086`. Precision tăng lên `1.000` nhưng vì chỉ có 18 TP và 385 FN nên model quá thận trọng và bỏ sót hầu hết xe. Recall theo kích thước cũng giảm: small `0.182` xuống `0.000`, medium `0.547` xuống `0.037`, large `0.561` xuống `0.171`. Ảnh `compare_round1.jpg` cho thấy điều này: ở `frame_0050`, cold start có TP/FP/FN `11/2/7` còn vòng 1 là `1/0/17`; ở `frame_0350`, kết quả là `9/2/14` so với `0/0/23`. Đây là bằng chứng model sau fine-tune kém hơn trên bộ tham chiếu, không phải bằng chứng nhãn người sửa kém hơn.

Quan sát độc lập trước pre-label là `BLIND_SCAN.md`: ở `frame_0099.jpg` đã đếm 26 xe và lưu ý các xe bị cắt ở mép ảnh. Khi rà nhãn, áp dụng guideline: xe chỉ thấy đèn nhưng còn đoán được thân vẫn gán một box ôm phần thân; không gán vệt sáng trên mặt đường; xe bị cắt chỉ vẽ phần nằm trong ảnh; hai xe sát nhau phải có hai box riêng. Một ca khó là xe tối chỉ còn đèn ở gần mép ảnh: vẫn gán phần thân có thể suy ra nhưng không kéo box vào vệt sáng trên mặt đường.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

So với cold start, vòng 1 tạo được bộ nhãn đã rà kỹ hơn ở mức dữ liệu: 204 box mới được thêm, 25 box được chỉnh và 16 box sai bị xóa. Tuy nhiên model fine-tune hiện kém hơn rõ rệt trên bộ tham chiếu: AP50 giảm `0.297`, recall giảm `0.444` và F1 giảm `0.554`. Vì vậy tôi dừng việc gán nhãn thêm để kiểm tra pipeline trước; không nên dùng thêm ảnh để che lấp lỗi khi model đang có 385 FN và recall chỉ `0.045`.

Hai ca có thể ưu tiên nếu làm vòng sau là `frame_0225.jpg` (hạng 44, U `0.9553`, A `0.5`, 9 box mơ hồ trên 35 box) và `frame_0372.jpg` (hạng 6, U `0.9202`, 15 box mơ hồ trên 42 box). Cần kiểm tra khoảng cách thời gian: `frame_0225.jpg` ở 90.0 giây gần `frame_0227.jpg` ở 90.8 giây, còn `frame_0372.jpg` ở 148.8 giây gần `frame_0369.jpg` ở 147.6 giây, nên không chọn cả các cặp gần trùng khi ngân sách rà nhãn hạn chế. Chi phí rà cao do mỗi ảnh có nhiều xe nhỏ, bị che hoặc chỉ thấy đèn.

Trước khi train thêm, cần kiểm tra đường dẫn và cấu trúc dữ liệu train, class id `0`, kích thước/độ rộng box, box trùng, quy tắc xe cắt mép, số box sau export và ngưỡng đánh giá. Cũng cần kiểm tra loss hoặc dự đoán sau train để phân biệt lỗi pipeline với overfit hoặc model quá thận trọng. Tập test chỉ có 20 ảnh, 14 box rất nhỏ bị bỏ qua và nhãn tham chiếu do model tạo chưa được người rà từng box; vì vậy metric phản ánh mức khớp với tham chiếu chứ không phải chân lý người. Tuy nhiên mức giảm lớn `0.297` AP50 và `0.444` recall vượt xa dao động nhỏ do tập test, nên phải xử lý như một cảnh báo thật trước khi tiếp tục.
