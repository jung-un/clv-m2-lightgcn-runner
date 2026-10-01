"""Small checks for the read-only support-backoff price-axis diagnostic."""

import numpy as np

import clv_m2_support_backoff_price_checkpoint_diagnostic as diagnostic


def test_comparisons_keep_price_and_id_only_contrasts_distinct():
    metrics = {
        'full': {'recall@10': 2.0, 'price_purchase_amount_weighted_hit@10': 4.0},
        'without_price': {'recall@10': 1.0, 'price_purchase_amount_weighted_hit@10': 5.0},
        'without_user_nv': {'recall@10': 1.5, 'price_purchase_amount_weighted_hit@10': 3.0},
        'id_only': {'recall@10': .5, 'price_purchase_amount_weighted_hit@10': 2.0},
    }
    frame = diagnostic._comparisons(metrics, {'recall@10': 1.0,
        'price_purchase_amount_weighted_hit@10': 2.0})
    price = frame[(frame.view == 'without_price') & (frame.reference == 'full')]
    assert set(price.metric) == set(metrics['full'])
    assert np.isclose(price.loc[price.metric == 'price_purchase_amount_weighted_hit@10',
                                'relative_change_pct'].iloc[0], 25.0)
    assert ((frame.view == 'id_only') & (frame.reference == 'without_price')).any()


if __name__ == '__main__':
    test_comparisons_keep_price_and_id_only_contrasts_distinct()
    print('support-backoff checkpoint diagnostic checks passed')
