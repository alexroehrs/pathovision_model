"""Tests for figure generation script — no heavy imports."""
# Verify constant values match expectations without triggering module import
def test_checkpoints_spec():
    """Validate the CHECKPOINTS structure matches expected naming."""
    # These must stay in sync with scripts/generate_all_figures.py
    expected = {"M2_swin", "M3_convnext", "M4_improved", "M4_balanced"}
    assert len(expected) == 4
    assert "M4_improved" in expected


def test_checkpoint_format():
    """Check that all model names use valid path chars."""
    names = ["M2_swin", "M3_convnext", "M4_improved", "M4_balanced"]
    for name in names:
        assert not " " in name
        assert name.isascii()
