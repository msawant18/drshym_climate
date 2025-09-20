from ingest.tiler import sliding_windows

def test_sliding_windows_count():
    wins = list(sliding_windows(1024, 1024, 512, 64))
    assert len(wins) > 0
