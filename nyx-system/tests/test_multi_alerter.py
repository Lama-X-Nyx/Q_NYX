"""
TDD Tests — MultiAlerter

Fan-out to N underlying alerters. One backend failing must NOT prevent
the others from being called, and the result is True if ANY backend
succeeded.
"""
import pytest


class _Recorder:
    """Minimal alerter double implementing send/info/warn/error/critical."""
    def __init__(self, name: str, fail: bool = False, return_false: bool = False):
        self.name = name
        self.fail = fail
        self.return_false = return_false
        self.calls = []

    def send(self, text):
        self.calls.append(('send', text))
        if self.fail:
            raise RuntimeError(f"{self.name} crashed")
        return not self.return_false

    def info(self, text):
        self.calls.append(('info', text))
        return not self.return_false

    def warn(self, text):
        self.calls.append(('warn', text))
        return not self.return_false

    def error(self, text):
        self.calls.append(('error', text))
        return not self.return_false

    def critical(self, text):
        self.calls.append(('critical', text))
        return not self.return_false


class TestFanout:

    def test_send_fans_out_to_all(self):
        from src.paper_live.multi_alerter import MultiAlerter
        a, b, c = _Recorder('a'), _Recorder('b'), _Recorder('c')
        m = MultiAlerter([a, b, c])
        m.send("hello")
        assert a.calls == [('send', "hello")]
        assert b.calls == [('send', "hello")]
        assert c.calls == [('send', "hello")]

    def test_returns_true_if_any_succeeds(self):
        from src.paper_live.multi_alerter import MultiAlerter
        a = _Recorder('a', return_false=True)  # disabled
        b = _Recorder('b')                     # ok
        m = MultiAlerter([a, b])
        assert m.send("x") is True

    def test_returns_false_if_all_fail(self):
        from src.paper_live.multi_alerter import MultiAlerter
        a = _Recorder('a', return_false=True)
        b = _Recorder('b', return_false=True)
        m = MultiAlerter([a, b])
        assert m.send("x") is False

    def test_exception_in_one_does_not_stop_others(self):
        """If backend A throws, B must still be called."""
        from src.paper_live.multi_alerter import MultiAlerter
        a = _Recorder('a', fail=True)
        b = _Recorder('b')
        m = MultiAlerter([a, b])
        ok = m.send("x")
        assert b.calls == [('send', "x")]
        assert ok is True   # b succeeded

    def test_empty_alerters_returns_false(self):
        from src.paper_live.multi_alerter import MultiAlerter
        m = MultiAlerter([])
        assert m.send("x") is False


class TestLevels:

    def test_info_fanout(self):
        from src.paper_live.multi_alerter import MultiAlerter
        a, b = _Recorder('a'), _Recorder('b')
        m = MultiAlerter([a, b])
        m.info("hi")
        assert a.calls == [('info', "hi")]
        assert b.calls == [('info', "hi")]

    def test_error_fanout(self):
        from src.paper_live.multi_alerter import MultiAlerter
        a, b = _Recorder('a'), _Recorder('b')
        m = MultiAlerter([a, b])
        m.error("bad")
        assert a.calls == [('error', "bad")]
        assert b.calls == [('error', "bad")]

    def test_critical_fanout(self):
        from src.paper_live.multi_alerter import MultiAlerter
        a, b = _Recorder('a'), _Recorder('b')
        m = MultiAlerter([a, b])
        m.critical("down")
        assert a.calls == [('critical', "down")]
        assert b.calls == [('critical', "down")]
