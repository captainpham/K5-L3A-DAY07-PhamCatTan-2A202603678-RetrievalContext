# Định dạng dữ liệu

## Dữ liệu bài thực hiện

Bài của Phạm Cát Tân dùng [corpus.json](corpus.json) để ghi nguồn, giấy phép, checksum và cấu hình index Day 6; [pages.json](pages.json) giữ 31 trang text; [chunks.json](chunks.json) giữ 135 chunk A–1.000; [benchmark.json](benchmark.json) chứa 17 câu dev và 7 câu test; [configs.json](configs.json) chứa 5 cấu hình thực nghiệm. Đây là các file dữ liệu thật của bài, khác với các file `example_*.json` bên dưới.

`benchmark.json` là JSON array. Câu có đáp án có `answerable: true`, `expected_answer` và `gold`; mỗi phần tử gold là một bằng chứng thay thế đúng nguồn/trang. Câu thiếu bằng chứng có `answerable: false`, `expected_behavior`, `absence_reason` và `absence_check`. Evidence được kiểm tra trên text trang gốc; việc excerpt có bị chia qua chunk hay không là kết quả cần đánh giá, không tự coi câu đó là thiếu bằng chứng. Metrics chuẩn hóa Unicode/khoảng trắng trước khi so khớp. Trace và nhãn đọc đối chiếu nằm trong [results/dev-final](../results/dev-final/) và [results/test-final](../results/test-final/).

## Định dạng mẫu của starter

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).
