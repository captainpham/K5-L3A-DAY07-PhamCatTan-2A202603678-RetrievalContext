"""Aggregate agent traces with human labels; never infer grounding from citation IDs."""
import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path


def aggregate(cases, traces, reviews):
    """Cases: id/answerable; traces: query_id/status/events/citations.

    Reviews: query_id, answer_grade (0..2 for answerable), citation_support
    (one boolean per citation), and abstention_correct (boolean for unanswerable).
    Provider errors are excluded and reported.
    """
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('case ID trùng')
    if len({r['query_id'] for r in traces}) != len(traces) or len({r['query_id'] for r in reviews}) != len(reviews):
        raise ValueError('trace/review ID trùng')
    trace_by_id = {r['query_id']: r for r in traces}
    review_by_id = {r['query_id']: r for r in reviews}
    out = {'cases': len(cases), 'provider_errors': 0, 'tool_use': {'n': 0, 'N': 0},
           'grounded_answer': {'0': 0, '1': 0, '2': 0, 'N': 0, 'mean': None},
           'citation_validity': {'n': 0, 'N': 0}, 'answer_citation_coverage': {'n': 0, 'N': 0},
           'abstention': {'n': 0, 'N': 0}, 'false_abstention': {'n': 0, 'N': 0}}
    for case in cases:
        cid = case['id']
        if not isinstance(case.get('answerable'), bool): raise ValueError(f'{cid}: thiếu answerable boolean')
        if cid not in trace_by_id: raise ValueError(f'{cid}: thiếu trace')
        t = trace_by_id[cid]
        if t.get('status') == 'error':
            out['provider_errors'] += 1
            continue
        if t.get('status') not in ('answered', 'abstain'): raise ValueError(f'{cid}: status không hợp lệ')
        if cid not in review_by_id: raise ValueError(f'{cid}: thiếu nhãn audit')
        r = review_by_id[cid]
        events = t.get('events')
        citations = t.get('citations')
        support = r.get('citation_support')
        if not isinstance(events, list) or not isinstance(citations, list): raise ValueError(f'{cid}: events/citations không hợp lệ')
        if not isinstance(support, list) or len(support) != len(citations) or any(type(v) is not bool for v in support):
            raise ValueError(f'{cid}: cần một nhãn boolean cho mỗi citation')
        out['tool_use']['N'] += 1
        out['tool_use']['n'] += any(e.get('event') == 'search' for e in events)
        out['citation_validity']['n'] += sum(support)
        out['citation_validity']['N'] += len(citations)
        if t['status'] == 'answered':
            out['answer_citation_coverage']['N'] += 1
            out['answer_citation_coverage']['n'] += bool(citations)
        if case['answerable']:
            grade = r.get('answer_grade')
            if type(grade) is not int or grade not in (0, 1, 2): raise ValueError(f'{cid}: answer_grade phải là 0, 1 hoặc 2')
            if t['status'] == 'abstain' and grade != 0: raise ValueError(f'{cid}: abstain trên câu có đáp án phải được chấm 0')
            out['grounded_answer'][str(grade)] += 1
            out['grounded_answer']['N'] += 1
            out['false_abstention']['N'] += 1
            out['false_abstention']['n'] += t['status'] == 'abstain'
        else:
            correct = r.get('abstention_correct')
            if type(correct) is not bool:
                raise ValueError(f'{cid}: cần nhãn abstention_correct do người chấm xác nhận')
            if correct and t['status'] != 'abstain':
                raise ValueError(f'{cid}: nhãn từ chối đúng không khớp status')
            out['abstention']['N'] += 1
            out['abstention']['n'] += correct
    g = out['grounded_answer']
    if g['N']: g['mean'] = (g['1'] + 2*g['2']) / g['N']
    return out



# Đánh giá retrieval/context dùng cùng trace và các nhãn đã đọc bên trên.
def canonical(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def evidence_hit(chunks, gold):
    # Các phần tử gold là bằng chứng thay thế hợp lệ, không phải đáp án đưa vào agent.
    return any(c['source_id'] == g['source_id'] and c['page'] == g['page']
               and canonical(g['evidence']) in canonical(c['text']) for c in chunks for g in gold)


def merged_chunks(chunks):
    """Ghép phần chồng/lân cận cùng trang; không tự điền nội dung nằm trong khoảng trống."""
    merged = []
    unique = {c['id']: c for c in chunks}
    rows = sorted(unique.values(), key=lambda c: (c['source_id'], c['page'], c['metadata']['char_start']))
    for row in rows:
        start, end = row['metadata']['char_start'], row['metadata']['char_end']
        if merged and (merged[-1]['source_id'], merged[-1]['page']) == (row['source_id'], row['page']):
            previous = merged[-1]
            if start <= previous['end']:
                previous['text'] += row['text'][max(0, previous['end'] - start):]
                previous['end'] = max(previous['end'], end)
                continue
        merged.append({**row, 'end': end})
    return merged


def retrieval_metrics(cases, traces):
    expected = {q['id']: q for q in cases}
    counts = {'hit1': 0, 'hit3': 0, 'context_evidence': 0, 'N': 0}
    for trace in traces:
        q = expected[trace['query_id']]
        if not q['answerable']:
            continue
        counts['N'] += 1
        ranking = trace.get('original_retrieval', {}).get('results', [])
        counts['hit1'] += evidence_hit(ranking[:1], q['gold'])
        counts['hit3'] += evidence_hit(ranking[:3], q['gold'])
        rows = {r['id']: r for e in trace.get('events', []) if e['event'] == 'search' for r in e['results']}
        context_rows = [rows[cid] for cid in trace.get('context_ids', []) if cid in rows]
        counts['context_evidence'] += evidence_hit(merged_chunks(context_rows), q['gold'])
    return {name: {'n': value, 'N': counts['N'],
                   'rate': value / counts['N'] if counts['N'] else None}
            for name, value in counts.items() if name != 'N'}


def summarize(cases, traces, reviews=None):
    summary = {'retrieval': retrieval_metrics(cases, traces),
               'total_cases': len(cases),
               'errors': sum(t['status'] == 'error' for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'search_call_distribution': {str(i): sum(t.get('search_calls', 0) == i for t in traces)
                                            for i in sorted({t.get('search_calls', 0) for t in traces})}}
    # Không biến việc khớp citation ID thành nhãn citation hỗ trợ nội dung.
    if reviews is not None:
        summary['agent'] = aggregate(cases, traces, reviews)
        by_id = {r['query_id']: r for r in reviews}
        total = sum((by_id[q['id']]['answer_grade'] / 2 if q['answerable']
                     else int(by_id[q['id']]['abstention_correct']))
                    for q in cases if next(t for t in traces if t['query_id'] == q['id'])['status'] != 'error')
        summary['selection_quality'] = total / len(cases)
        grades = [by_id[q['id']]['context_grade'] for q in cases if q['answerable'] and q['id'] in by_id]
        summary['context_review'] = {str(g): grades.count(g) for g in (0, 1, 2)}
    costs = [c.get('usage', {}).get('cost') for t in traces for c in t.get('model_calls', [])]
    summary['chat_cost_usd_reported'] = sum(c for c in costs if isinstance(c, (int, float)))
    summary['chat_calls_without_cost'] = sum(c is None for c in costs)
    summary['embedding_usage'] = {
        key: sum(t.get('embedding_usage', {}).get(key, 0) for t in traces)
        for key in ('api_requests', 'cache_hits', 'prompt_tokens', 'reported_cost_usd', 'responses_without_cost')}
    return summary


def main():
    parser = argparse.ArgumentParser(description='Tổng hợp trace agent và nhãn audit thủ công')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def load(path): return json.loads(path.read_text(encoding='utf-8'))
    cases = [c for c in load(args.cases) if c.get('split') == args.split]
    if not cases: raise ValueError('không có case trong split')
    report = {'split': args.split, **aggregate(cases, load(args.traces), load(args.reviews))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
