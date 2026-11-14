"""CLI: inspect, ask, benchmark và select. Dùng --help để xem tham số chạy."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from starter.agentic import Controller, Retriever, run_agent, select_context
from starter.agent_metrics import canonical, summarize
from .prepare import read_lines, write_json
from .settings import ROOT, digest, load_settings, read_json


def source_hash():
    # Bất kỳ thay đổi pipeline sau lúc chốt đều yêu cầu chốt lại trên dev.
    paths = sorted((ROOT / 'src').glob('*.py')) + [ROOT / 'starter/agentic.py', ROOT / 'starter/baseline.py', ROOT / 'starter/agent_metrics.py']
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def validate_cases(cases, split):
    """Kiểm tra bằng chứng trên trang gốc, không ép phải nằm trong một chunk."""
    pages = {(p['document_id'], p['page']): p for p in read_json(ROOT / 'data/pages.json')}
    selected = [q for q in cases if q['split'] == split]
    if len({q['id'] for q in selected}) != len(selected):
        raise ValueError('ID câu hỏi trùng.')
    minimum = (15, 2) if split == 'dev' else (5, 2)
    if sum(q['answerable'] for q in selected) < minimum[0] or sum(not q['answerable'] for q in selected) < minimum[1]:
        raise ValueError('Chưa đủ số câu có/thiếu bằng chứng.')
    for q in selected:
        if not q.get('question', '').strip() or type(q.get('answerable')) is not bool:
            raise ValueError('Câu hỏi/answerable không hợp lệ.')
        if q['answerable']:
            if not q.get('gold') or not q.get('expected_answer'):
                raise ValueError(f"{q['id']}: thiếu gold/đáp án.")
            for g in q['gold']:
                page = pages.get((g['source_id'], g['page']))
                if not page or not g['evidence'].strip() or canonical(g['evidence']) not in canonical(page['text']):
                    raise ValueError(f"{q['id']}: evidence không có trên trang gốc.")
                if any(page['metadata'].get(k) != v for k, v in (q.get('filter') or {}).items()):
                    raise ValueError(f"{q['id']}: evidence trái phạm vi filter.")
        elif not q.get('absence_reason') or not q.get('absence_check'):
            raise ValueError(f"{q['id']}: thiếu kiểm chứng thông tin vắng mặt.")
    return selected


def public_identity(settings):
    # Không xuất DSN hoặc API key vào artifact.
    return {'namespace': settings['namespace'], 'strategy': settings['strategy'],
            'embedding': settings['embedding'], 'dimensions': settings['dimensions'],
            'agent_model': settings['agent_model']}


def make_retriever(settings, config):
    retriever = Retriever(settings, config['method'], config['candidate_k'])
    expected = read_json(ROOT / 'data/chunks.json')
    if digest(retriever.chunks) != digest(sorted(expected, key=lambda c: c['id'])):
        raise ValueError('Dữ liệu PostgreSQL đã khác snapshot benchmark; dừng để tránh chấm sai.')
    return retriever


def run_case(case, settings, config, retriever, retrieval_only=False):
    tick = time.perf_counter()
    before = dict(retriever.embedder.stats)
    # Đo ranking query gốc riêng; không cho model thấy gold hoặc nhãn answerable.
    try:
        retriever.search(case['question'], filters=case.get('filter'), top_k=config['top_k'])
        original = retriever.last_trace
    except Exception as exc:
        return {'query_id': case['id'], 'status': 'error', 'error_type': type(exc).__name__,
                'events': [], 'citations': [], 'context_ids': [], 'answer': '',
                'elapsed_ms': (time.perf_counter() - tick) * 1000, 'search_calls': 0}
    if retrieval_only:
        rows = original['results']
        context, ids = select_context([rows], config['budget'])
        result = {'status': 'retrieval_only', 'events': [{'event': 'search', 'results': rows}],
                  'context': context, 'context_ids': ids, 'citations': [], 'search_calls': 1}
    else:
        controller = Controller(settings, config['max_searches'])
        result = run_agent(case['question'], controller, retriever, filters=case.get('filter'), capture_errors=True,
                           **{k: config[k] for k in ('max_searches', 'top_k', 'budget', 'recent_first')})
        result['model_calls'] = controller.calls
    result.update(query_id=case['id'], question=case['question'], original_retrieval=original,
                  elapsed_ms=(time.perf_counter() - tick) * 1000,
                  embedding_usage={k: retriever.embedder.stats[k] - before[k] for k in before})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inspect', help='Kiểm tra snapshot, model và benchmark; không gọi API.')
    ask = commands.add_parser('ask', help='Hỏi agent thật, lưu đầy đủ trace.')
    ask.add_argument('question')
    ask.add_argument('--config', default='hybrid_multiturn')
    bench = commands.add_parser('benchmark', help='Chạy dev/test và lưu kết quả từng câu.')
    bench.add_argument('--split', choices=['dev', 'test'], default='dev')
    bench.add_argument('--configs', nargs='+')
    bench.add_argument('--retrieval-only', action='store_true')
    bench.add_argument('--selection', type=Path)
    bench.add_argument('--output', type=Path, required=True)
    select = commands.add_parser('select', help='Chốt từ kết quả dev và nhãn đã đọc, trước test.')
    select.add_argument('--run', type=Path, required=True)
    select.add_argument('--reviews', type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings()
    configs = read_json(ROOT / 'data/configs.json')
    cases = read_json(ROOT / 'data/benchmark.json')
    identity = {**public_identity(settings), 'code_sha256': source_hash(),
                'benchmark_sha256': digest(cases), 'configs_sha256': digest(configs),
                'chunks_sha256': digest(read_json(ROOT / 'data/chunks.json'))}
    if args.command == 'inspect':
        retriever = make_retriever(settings, configs['baseline'])
        for split in ('dev', 'test'):
            validate_cases(cases, split)
        print(json.dumps({**public_identity(settings), 'chunks': len(retriever.chunks),
                          'api_key_configured': bool(settings['api_key']),
                          'benchmark_cases': len(cases)}, ensure_ascii=False, indent=2))
        return
    if args.command == 'ask':
        config = configs[args.config]
        retriever = make_retriever(settings, config)
        result = run_case({'id': 'interactive', 'question': args.question}, settings, config, retriever)
        path = ROOT / 'results' / ('ask-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        write_json(path, {**identity, 'config': config, 'trace': result})
        print(result.get('answer') or f"Lỗi: {result.get('error_type')}")
        print('Status:', result['status'], '| Trace:', path)
        return
    if args.command == 'select':
        meta = read_json(args.run / 'run.json')
        if meta['split'] != 'dev' or meta['retrieval_only'] or meta['identity'] != identity:
            raise ValueError('Chỉ chọn từ agent dev với code/dữ liệu/model hiện tại.')
        reviews = read_json(args.reviews)
        summaries = {}
        dev_cases = validate_cases(cases, 'dev')
        for name in meta['configs']:
            traces = read_lines(args.run / f'{name}.jsonl')
            if {t['query_id'] for t in traces} != {q['id'] for q in dev_cases}:
                raise ValueError(f'{name}: chưa đủ trace dev.')
            labels = [r for r in reviews if r['config'] == name]
            for label in labels:
                if not label.get('reason') or not label.get('reviewer') or type(label.get('context_grade')) is not int or label['context_grade'] not in (0, 1, 2):
                    raise ValueError('Nhãn phải có lý do, người/agent review và context_grade 0–2.')
            summaries[name] = summarize(dev_cases, traces, labels)
        # Tiêu chí cố định: chất lượng đủ cả answer/abstention, rồi context, ít lỗi, thời gian.
        winner = min(summaries, key=lambda n: (-summaries[n]['selection_quality'],
                         -summaries[n]['retrieval']['context_evidence']['rate'],
                         summaries[n]['errors'], summaries[n]['median_ms'], n))
        selection = {'selected_on': 'dev', 'selected_config': winner, 'identity': identity,
                     'created_at_utc': datetime.now(timezone.utc).isoformat(),
                     'criteria': 'answer_grade/2 + correct_abstention over all cases desc; context evidence desc; errors asc; median_ms asc; name asc',
                     'dev_run': str(args.run.resolve()), 'reviews_sha256': digest(reviews), 'summaries': summaries}
        write_json(args.run / 'selection.json', selection)
        print('Selected:', winner)
        return
    selected_cases = validate_cases(cases, args.split)
    names = args.configs or list(configs)
    if args.split == 'test':
        if not args.selection:
            raise ValueError('Test cần --selection được chốt từ dev.')
        selection = read_json(args.selection)
        if selection['selected_on'] != 'dev' or selection['identity'] != identity:
            raise ValueError('Code/dữ liệu/model đã khác lúc chốt dev.')
        names = list(dict.fromkeys(['baseline', selection['selected_config']]))
    if not all(name in configs for name in names):
        raise ValueError('Tên cấu hình không hợp lệ.')
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output / 'run.json'
    meta = {'identity': identity, 'split': args.split, 'configs': names,
            'retrieval_only': args.retrieval_only, 'configurations': {n: configs[n] for n in names},
            'selection_criteria': 'answer_grade/2 + correct_abstention over all cases desc; context evidence desc; errors asc; median_ms asc; name asc'}
    if meta_path.exists() and read_json(meta_path) != meta:
        raise ValueError('Thư mục output thuộc cấu hình khác; chọn thư mục mới.')
    write_json(meta_path, meta)
    summaries = {}
    for name in names:
        retriever = make_retriever(settings, configs[name])
        path = args.output / f'{name}.jsonl'
        traces = read_lines(path) if path.exists() else []
        done = {t['query_id'] for t in traces}
        for case in selected_cases:
            if case['id'] in done:
                continue
            result = run_case(case, settings, configs[name], retriever, args.retrieval_only)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            traces.append(result)
            print(name, case['id'], result['status'], flush=True)
        summaries[name] = summarize(selected_cases, traces)
        write_json(args.output / 'summary.json', summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
