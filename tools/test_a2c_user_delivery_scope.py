import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import a2c_user_delivery_scope as scope

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'runs/a2c_research_20261020/final_evaluation/delivery_scope_31_jobs.json'


class DeliveryScopeTests(unittest.TestCase):
    def test_exact_human_scope_metadata_passes(self):
        value = scope.load_scope(ROOT, PATH)
        self.assertEqual(len(value['jobs']), 31)
        self.assertEqual(len(value['frozen']['jobs']), 32)
        self.assertNotIn(scope.DEFERRED, [job['name'] for job in value['jobs']])

    def rejected(self, change):
        original = scope.read
        altered = copy.deepcopy(original(PATH))
        change(altered)
        with patch.object(scope, 'read', side_effect=lambda path: altered if Path(path).resolve() == PATH else original(path)):
            with self.assertRaises(ValueError):
                scope.load_scope(ROOT, PATH)

    def test_cannot_drop_a_candidate(self):
        self.rejected(lambda value: value['included_jobs'].pop(0))

    def test_cannot_restore_deferred_reference(self):
        self.rejected(lambda value: value['included_jobs'].append(scope.DEFERRED))

    def test_cannot_reduce_remaining_game_budget(self):
        self.rejected(lambda value: value.update(included_games=6000))

    def test_cannot_change_policy_selection(self):
        self.rejected(lambda value: value.update(policy_or_checkpoint_selection_changed=True))

    def test_cannot_forge_user_request_hash(self):
        self.rejected(lambda value: value.update(user_request_sha256='0'*64))

    def test_cannot_select_another_omitted_job(self):
        self.rejected(lambda value: value.update(deferred_jobs=['residual_seed11_guarded_renju_greedy']))


if __name__ == '__main__':
    unittest.main()
