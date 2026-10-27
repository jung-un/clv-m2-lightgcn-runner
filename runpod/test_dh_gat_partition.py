import json
import unittest

import dh_gat_partition as p


class PartitionTests(unittest.TestCase):
    def manifest(self):
        return dict(state='l4_restricted_and_resumed', l4_seeds=list(p.L4_SEEDS),
            ada_seeds=list(p.ADA_SEEDS), source=p.platform.JOBS['dh-gat']['pin'],
            launcher_sha256=p.OLD_LAUNCHER_SHA, wrapper_sha256=p.platform.sha(p.__file__),
            checkpoint_epoch=186, resumed_epoch=187)

    def test_disjoint_complete_partition(self):
        self.assertFalse(set(p.L4_SEEDS) & set(p.ADA_SEEDS))
        self.assertEqual(set(p.L4_SEEDS + p.ADA_SEEDS), set(p.OLD_SEEDS) - {42})
        self.assertFalse({42,49} & set(p.ADA_SEEDS))

    def test_manifest_requires_actual_resumption_and_fixed_ownership(self):
        p.validate_manifest(self.manifest())
        for key, value in [('state','waiting'), ('resumed_epoch',186),
                           ('ada_seeds',[43]), ('source','changed')]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                p.validate_manifest({**self.manifest(),key:value})

    def test_paths_and_json_tuple_representation_do_not_change_science(self):
        old = {key:{} for key in p.IDENTITY_KEYS}
        old['config'] = dict(epochs=300,seeds=(42,43),out_dir='old',seed49_reference='old')
        new = json.loads(json.dumps(old))
        new['config'].update(out_dir='new',seed49_reference='new')
        self.assertEqual(p.scientific_identity(old),p.scientific_identity(new))
        new['config']['epochs'] = 299
        self.assertNotEqual(p.scientific_identity(old),p.scientific_identity(new))

    def checkpoint(self):
        return dict(identity=dict(seed=43,config_hash=p.RUN_ID),epoch=3,
            history=[dict(epoch=i) for i in range(1,4)],model_state={},optimizer_state={},
            numpy_rng_state={},torch_rng_state=[],cuda_rng_state=[[]])

    def test_checkpoint_requires_contiguous_history_rng_identity_and_hash(self):
        progress = dict(epoch=3,checkpoint_sha256='verified')
        self.assertEqual(p.validate_checkpoint(self.checkpoint(),progress,'verified')[0],3)
        for key,value in [('history',[]),('cuda_rng_state',[]),('epoch',300),
                          ('identity',dict(seed=46,config_hash=p.RUN_ID))]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                p.validate_checkpoint({**self.checkpoint(),key:value},progress,'verified')
        with self.assertRaises(RuntimeError):
            p.validate_checkpoint(self.checkpoint(),progress,'corrupt')


if __name__ == '__main__':
    unittest.main()
