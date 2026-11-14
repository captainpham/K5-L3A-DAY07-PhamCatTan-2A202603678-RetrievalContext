"""Kiểm tra giới hạn, context và chống citation ngoài context của agent Day 7."""
import unittest

from starter.agentic import run_agent
from src.run import validate_cases
from src.settings import ROOT, read_json


class Controller:
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, question, history):
        return next(self.actions)


class Retriever:
    last_trace = {}

    def search(self, query, *, filters, top_k):
        return [{'id': 'c1', 'text': 'Thông tin có thật.', 'source': 'doc.pdf', 'page': 1}]


class ProductionAgentTests(unittest.TestCase):
    def test_blocks_extra_inline_citation_not_declared_in_list(self):
        fake = 'f' * 64
        controller = Controller([{'action': 'search', 'query': 'q'},
                                 {'action': 'answer', 'answer': f'Đáp án [c1] [{fake}]', 'citations': ['c1']}])
        result = run_agent('q', controller, Retriever())
        self.assertEqual(result['stop_reason'], 'invalid_citations')

    def test_mathematical_brackets_are_not_mistaken_for_citation_ids(self):
        controller = Controller([{'action': 'search', 'query': 'q'},
                                 {'action': 'answer', 'answer': 'Công thức E[π] [c1]', 'citations': ['c1']}])
        self.assertEqual(run_agent('q', controller, Retriever())['status'], 'answered')

    def test_explicit_abstention_keeps_reason(self):
        controller = Controller([{'action': 'search', 'query': 'q'},
                                 {'action': 'abstain', 'answer': 'Không có thông tin điện năng trong tài liệu.',
                                  'citations': []}])
        result = run_agent('q', controller, Retriever())
        self.assertEqual(result['status'], 'abstain')
        self.assertIn('điện năng', result['answer'])

    def test_error_preserves_search_trace_without_leaking_backend_message(self):
        class Broken:
            calls = 0

            def decide(self, question, history):
                self.calls += 1
                if self.calls == 1:
                    return {'action': 'search', 'query': 'q'}
                raise RuntimeError('secret-token')
        result = run_agent('q', Broken(), Retriever(), capture_errors=True)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(result['events']), 1)
        self.assertNotIn('secret-token', str(result))

    def test_search_limit_does_not_make_extra_retrieval_call(self):
        controller = Controller([{'action': 'search', 'query': 'q'}, {'action': 'search', 'query': 'again'}])
        result = run_agent('q', controller, Retriever(), max_searches=1)
        self.assertEqual(result['search_calls'], 1)
        self.assertEqual(result['stop_reason'], 'search_limit')

    def test_benchmark_gold_exists_in_original_pages_and_split_is_complete(self):
        cases = read_json(ROOT / 'data/benchmark.json')
        self.assertEqual(len(validate_cases(cases, 'dev')), 17)
        self.assertEqual(len(validate_cases(cases, 'test')), 7)


if __name__ == '__main__':
    unittest.main()
