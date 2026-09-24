import time

from app.ingest import thermal


def test_reads_a_plausible_cpu_temperature():
    t = thermal.cpu_package_temp()
    assert t is None or 10 < t < 110


def test_duty_cycle_idles_in_proportion_to_work(monkeypatch):
    monkeypatch.setattr(thermal, "cpu_package_temp", lambda: 50.0)
    g = thermal.ThermalGuard(duty=0.5, threads=2, check_every=1e9)
    time.sleep(0.1)  # "work"
    t0 = time.time()
    g.step()
    assert 0.08 < time.time() - t0 < 0.3  # idles about as long as it worked


def test_pauses_when_hot_until_cool(monkeypatch):
    temps = iter([95.0, 93.0, 94.0, 90.0, 91.0, 89.0, 70.0, 75.0, 72.0])
    monkeypatch.setattr(thermal, "cpu_package_temp", lambda: next(temps))
    monkeypatch.setattr(thermal.time, "sleep", lambda s: None)
    g = thermal.ThermalGuard(duty=1.0, threads=2, check_every=0)
    g.step()  # median 94 -> pause; median 90 still hot; median 72 resumes
    assert next(temps, None) is None  # all three rounds of samples consumed


def test_a_momentary_spike_does_not_pause(monkeypatch):
    temps = iter([99.0, 70.0, 72.0])
    monkeypatch.setattr(thermal, "cpu_package_temp", lambda: next(temps))
    monkeypatch.setattr(thermal.time, "sleep", lambda s: None)
    g = thermal.ThermalGuard(duty=1.0, threads=2, check_every=0)
    g.step()
    assert g.paused_seconds == 0.0
