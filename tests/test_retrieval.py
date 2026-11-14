"""Kiểm tra fusion/filter bằng fixture; không dùng các điểm này làm benchmark thật."""
import unittest

from starter.agent_metrics import evidence_hit, merged_chunks
from starter.agentic import checked_filters, rrf


def chunk(cid, text, start=0):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf',
            'page': 1, 'score': 0.8, 'metadata': {'char_start': start, 'char_end': start + len(text)}}


class RetrievalTests(unittest.TestCase):
    def test_rrf_uses_ranks_instead_of_incompatible_raw_scores(self):
        dense = [chunk('a', 'first'), chunk('b', 'second')]
        lexical = [{**chunk('b', 'second'), 'score': 900}, {**chunk('c', 'third'), 'score': 800}]
        result = rrf({'dense': dense, 'tfidf': lexical})
        self.assertEqual(result[0]['id'], 'b')
        self.assertAlmostEqual(result[0]['score'], 1 / 62 + 1 / 61)
        self.assertEqual(len({r['id'] for r in result}), 3)

    def test_filter_rejects_gold_source_and_nested_values(self):
        for filters in ({'gold_page': 7}, {'source_id': 'doc'}, {'section_group': ['main']}):
            with self.assertRaises(ValueError):
                checked_filters(filters)
        self.assertEqual(checked_filters({'section_group': 'main'}), {'section_group': 'main'})

    def test_context_merges_overlap_without_inventing_missing_text(self):
        gold = [{'source_id': 'doc', 'page': 1, 'evidence': 'alpha beta gamma'}]
        rows = [chunk('a', 'alpha beta'), chunk('b', 'beta gamma', 6)]
        self.assertFalse(evidence_hit(rows, gold))
        self.assertTrue(evidence_hit(merged_chunks(rows), gold))
        self.assertFalse(evidence_hit(merged_chunks([rows[0], chunk('c', 'gamma', 30)]), gold))


if __name__ == '__main__':
    unittest.main()
