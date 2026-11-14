"""Chuẩn bị corpus/benchmark có nguồn; chỉ đọc snapshot và không sửa Day 6."""
import json
from collections import Counter

from .settings import ROOT, digest, load_settings, read_json


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare():
    settings = load_settings()
    snapshot = settings['day6'] / 'results/20261113T124446335645Z'
    pages = read_lines(snapshot / 'pages.jsonl')
    chunks = read_lines(snapshot / 'chunks_A.jsonl')
    manifest = read_json(settings['day6'] / 'data/manifest.json')
    write_json(ROOT / 'data/corpus.json', {
        'namespace': settings['namespace'], 'strategy': 'A', 'dimensions': settings['dimensions'],
        'embedding': settings['embedding'], 'pages': len(pages), 'chunks': len(chunks),
        'documents': manifest['documents'], 'snapshot_sha256': digest(chunks),
        'page_numbering': '1-based physical PDF page',
    })
    write_json(ROOT / 'data/pages.json', pages)
    write_json(ROOT / 'data/chunks.json', [
        {'id': c['chunk_id'], 'source_id': c['document_id'],
         **{k: c[k] for k in ('source', 'page', 'text', 'metadata')}} for c in chunks])
    original = read_json(settings['day6'] / 'data/benchmark.json')['questions']
    cases = []
    # Chỉ tái sử dụng dev Day 6; toàn bộ test Day 7 dùng câu mới về đáp án khác.
    for q in original:
        if q['split'] != 'dev':
            continue
        cases.append({'id': q['id'], 'split': 'dev', 'question': q['question'],
                      'answerable': True, 'expected_answer': q['expected_answer'],
                      'filter': q.get('filter'), 'gold': [
                          {'source_id': g['gold_source_id'], 'page': g['gold_page'],
                           'evidence': g['evidence_quote']}
                          for g in [q, *q.get('alternative_gold', [])]],
                      'provenance': 'Day 6 dev; không dùng lại test Day 6.'})
    # Alternative đã đọc đối chiếu: trang 3 cũng nêu đầy đủ ba tiêu chí của Q01.
    page3 = next(p['text'] for p in pages if p['document_id'] == '2605.17456v1' and p['page'] == 3)
    start = page3.index('A formalization of MIL evidence quality')
    end = page3.index('with', start)
    cases[0]['gold'].append({'source_id': '2605.17456v1', 'page': 3,
                            'evidence': page3[start:end].rstrip(' ,\n')})
    specs = [
        ('T01', '2605.17456v1', 4,
         'Trong phần P1 của GCE-MIL, giữ lại attention top-k làm Macro-F1 giảm bao nhiêu so với full bag, và con số đó được lấy trung bình trên phạm vi nào?',
         'Giảm 0.078 Macro-F1, trung bình trên 9 dataset và 9 backbone.',
         'Table 4 shows this fails:', 'nine datasets and nine backbones.'),
        ('T02', '2605.17456v1', 6,
         'Trong công thức patch-anchor response của GCE-MIL, gamma và delta có giá trị nào và mỗi tham số có tác dụng gì?',
         'Gamma = 8.0 làm sắc response; delta = 0.15 loại bớt các khớp yếu.',
         'where am is the frozen anchor embedding', 'weak matches.'),
        ('T03', '2605.17456v1', 8,
         'Theo phần Main Classification Results của GCE-MIL, trên BRACS, Macro-F1 của ABMIL tăng từ bao nhiêu lên bao nhiêu khi thêm GCE?',
         'Từ 0.634 lên 0.703, tăng 6.9 điểm phần trăm.',
         'On BRACS, ABMIL improves', 'Macro-F1 (+6.9 points)'),
        ('T04', '2606.06224v2', 7,
         'Trong phân tích symbolic representation space của Symb-xMIL trên các tumor sample, K-means dùng K bao nhiêu và silhouette bằng bao nhiêu?',
         'K = 4 và silhouette = 0.44.',
         'we appliedK-means clustering', 'silhouette = 0.44)'),
        ('T05', '2606.06224v2', 4,
         'Ở mục 3.2, Symb-xMIL thay đổi việc so sánh với logical query như thế nào so với cách dùng Harsanyi dividends?',
         'So sánh trực tiếp prediction f(x_S) với lambda_q(S), không áp dụng phép biến đổi Harsanyi dividends.',
         'We therefore propose a modification', 'without applying this transforma-\ntion.'),
    ]
    for cid, sid, number, question, answer, first, last in specs:
        text = next(p['text'] for p in pages if p['document_id'] == sid and p['page'] == number)
        start = text.index(first)
        end = text.index(last, start) + len(last)
        cases.append({'id': cid, 'split': 'test', 'answerable': True, 'question': question,
                      'expected_answer': answer, 'gold': [{'source_id': sid, 'page': number,
                                                         'evidence': text[start:end]}],
                      'filter': None, 'provenance': 'Câu mới Day 7, cố định trước chạy dev/test.'})
    absent = [
        ('D16', 'dev', 'GCE-MIL có độ trễ suy luận trung vị bao nhiêu mili giây trên CPU Intel Core i5-12400?',
         ['i5-12400'], 'Không có benchmark độ trễ trên CPU Intel Core i5-12400 trong corpus.'),
        ('D17', 'dev', 'Triển khai Symb-xMIL trên AWS us-east-1 tốn bao nhiêu USD mỗi tháng theo paper?',
         ['us-east-1'], 'Paper không báo giá triển khai AWS us-east-1 theo tháng.'),
        ('T06', 'test', 'Theo paper, mã hồ sơ đăng ký thử nghiệm lâm sàng ClinicalTrials.gov của GCE-MIL là gì?',
         ['clinicaltrials.gov', 'NCT0'], 'Không có hồ sơ hoặc mã đăng ký ClinicalTrials.gov trong corpus.'),
        ('T07', 'test', 'Theo paper Symb-xMIL, huấn luyện mô hình tiêu thụ bao nhiêu kWh điện?',
         ['kwh', 'kilowatt'], 'Không báo điện năng huấn luyện theo kWh trong corpus.'),
    ]
    for cid, split, question, terms, reason in absent:
        occurrences = [{'source_id': p['document_id'], 'page': p['page'], 'term': term}
                       for p in pages for term in terms if term.casefold() in p['text'].casefold()]
        if occurrences:
            raise ValueError(f'Cần xem lại câu thiếu bằng chứng {cid}: {occurrences}')
        cases.append({'id': cid, 'split': split, 'question': question, 'answerable': False,
                      'expected_behavior': 'search_and_abstain', 'absence_reason': reason,
                      'absence_check': {'scope': 'Toàn bộ 31 trang text trong corpus',
                                        'searched_terms': terms, 'occurrences': occurrences,
                                        'review': 'Đối chiếu phạm vi thông tin được hỏi; không suy từ top-k rỗng.'}})
    write_json(ROOT / 'data/benchmark.json', cases)
    print('Prepared:', len(pages), 'pages;', len(chunks), 'chunks;',
          dict(Counter((q['split'], q['answerable']) for q in cases)))


if __name__ == '__main__':
    prepare()
