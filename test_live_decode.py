import pytest
from live_decode import parse_args_custom, KEY_MAP

def test_parse_args_simple():
    argv = ['--visualize', '0']
    custom, remaining = parse_args_custom(argv)
    assert custom == {'set_ctrl': {}, 'hotkeys': {}}
    assert remaining == ['--visualize', '0']

def test_parse_set_ctrl():
    argv = ['-c', 'pan_absolute=100', '-c', 'zoom_absolute=20']
    custom, remaining = parse_args_custom(argv)
    assert custom['set_ctrl'] == {'pan_absolute': 100, 'zoom_absolute': 20}
    assert remaining == []

def test_parse_hotkeys():
    argv = ['-c', 'pan_absolute=100', '-k', 'left=-10', '-k', 'right=10']
    custom, remaining = parse_args_custom(argv)
    assert custom['set_ctrl'] == {'pan_absolute': 100}
    assert custom['hotkeys'] == {
        KEY_MAP['left']: ('pan_absolute', -10),
        KEY_MAP['right']: ('pan_absolute', 10)
    }
    assert remaining == []

def test_parse_hotkeys_for_multiple_ctrls():
    argv = [
        '-c', 'pan_absolute=100', '-k', 'left=-10',
        '-c', 'zoom_absolute=20', '-k', 'up=1', '-k', 'down=-1'
    ]
    custom, remaining = parse_args_custom(argv)
    assert custom['set_ctrl'] == {'pan_absolute': 100, 'zoom_absolute': 20}
    assert custom['hotkeys'] == {
        KEY_MAP['left']: ('pan_absolute', -10),
        KEY_MAP['up']: ('zoom_absolute', 1),
        KEY_MAP['down']: ('zoom_absolute', -1)
    }
    assert remaining == []

def test_parse_mixed_args():
    argv = ['--visualize', '-c', 'pan_absolute=100', '0', '-k', 'left=-10']
    custom, remaining = parse_args_custom(argv)
    assert custom['set_ctrl'] == {'pan_absolute': 100}
    assert custom['hotkeys'] == { KEY_MAP['left']: ('pan_absolute', -10) }
    assert remaining == ['--visualize', '0']

def test_invalid_ctrl_format():
    argv = ['-c', 'pan_absolute100']
    custom, remaining = parse_args_custom(argv)
    assert custom == {'set_ctrl': {}, 'hotkeys': {}}
    assert remaining == []

def test_invalid_hotkey_format():
    argv = ['-c', 'pan_absolute=100', '-k', 'left-10']
    custom, remaining = parse_args_custom(argv)
    assert custom['set_ctrl'] == {'pan_absolute': 100}
    assert custom['hotkeys'] == {}
    assert remaining == []
