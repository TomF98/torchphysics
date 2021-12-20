import torch
import pytest
import torchphysics as tp

def test_dataloader():
    points_a = tp.spaces.Points(torch.rand(10, 1), tp.spaces.R1('a'))
    points_b = tp.spaces.Points(torch.rand(10, 1), tp.spaces.R1('b'))
    points_c = tp.spaces.Points(torch.rand(10, 1), tp.spaces.R1('c'))
    loader = tp.utils.PointsDataLoader((points_a, points_b, points_c),
                                        batch_size=3,
                                        drop_last=False,
                                        shuffle=False)
    i = 0
    for batch in loader:
        i += 1
        a, b, c = batch
        assert len(a) == 3 or len(a) == 1
        assert len(b) == 3 or len(b) == 1
        assert len(c) == 3 or len(c) == 1
        assert a.space == tp.spaces.R1('a')
        assert b.space == tp.spaces.R1('b')
        assert c.space == tp.spaces.R1('c')
    assert i == 4