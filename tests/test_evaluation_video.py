import pytest

from sunerf.evaluation.video import _select_channel_metadata


def _channels():
    return (
        {'id': 'AIA_94', 'cmap': 'sdoaia94'},
        {'id': 'AIA_171', 'cmap': 'sdoaia171'},
        {'id': 'EUI_174', 'cmap': 'gray'},
    )


def test_video_channel_selection_resolves_aliases_and_preserves_requested_order():
    selected = _select_channel_metadata(_channels(), ['174', 'AIA_94'])

    assert tuple(channel['id'] for channel in selected) == ('EUI_174', 'AIA_94')


def test_video_channel_selection_rejects_unknown_duplicate_and_ambiguous_aliases():
    with pytest.raises(ValueError, match='Unknown channel'):
        _select_channel_metadata(_channels(), ['193'])
    with pytest.raises(ValueError, match='selected more than once'):
        _select_channel_metadata(_channels(), ['94', 'AIA_94'])
    with pytest.raises(ValueError, match='ambiguous'):
        _select_channel_metadata(
            _channels() + ({'id': 'EUVI_174', 'cmap': 'gray'},), ['174']
        )
