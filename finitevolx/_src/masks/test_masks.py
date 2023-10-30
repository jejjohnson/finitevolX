import pytest
from finitevolx._src.masks.utils import init_mask_rect, init_mask_realish

@pytest.fixture()
def mask_rectangular():
    mask = init_mask_rect(n=6, init="node")
    return mask



@pytest.fixture()
def mask_realish():
    return init_mask_realish(n=6, init="center")