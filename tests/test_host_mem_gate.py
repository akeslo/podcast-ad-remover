from app.core.processor import host_mem_used_fraction


def test_host_mem_used_fraction(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       1000 kB\nMemFree: 10 kB\nMemAvailable:    200 kB\n")
    assert abs(host_mem_used_fraction(str(f)) - 0.8) < 1e-9
    assert host_mem_used_fraction(str(tmp_path / "missing")) == 0.0
