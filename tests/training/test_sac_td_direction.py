import numpy as np
import pytest
from scripts.diagnose_sac_td_direction import episode_suffix, recorded_returns


def test_recorded_return_uses_real_tick_discounts_and_no_terminal_bootstrap():
    assert recorded_returns([1., -20.], [2, 1]) == pytest.approx([1-20*.99**2,-20])


def test_suffix_does_not_cross_episodes_or_missing_ticks():
    a=dict(episode_uid=np.array(['a','a','b','b','b']),tick=np.array([0,1,0,1,2]),
           delta_ticks=np.ones(5,int),bootstrap=np.array([True,False,True,True,False]),
           terminated=np.array([False,True,False,False,True]),truncated=np.zeros(5,bool))
    assert episode_suffix(a,4).tolist()==[2,3,4]
    with pytest.raises(ValueError): episode_suffix(a,3)
    a['tick'][3]=7
    with pytest.raises(ValueError,match='contiguous'): episode_suffix(a,4)
