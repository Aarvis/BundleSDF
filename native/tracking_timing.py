"""Synchronized wall timings and durable stage progress for pose-only tracking."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import time


class FrameTimings:
    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.active = False
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.counts = defaultdict(int)

    def begin(self, frame_id, callback=None):
        self.frame_id = frame_id
        self.callback = callback
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.counts = defaultdict(int)
        self.stack = []
        self.active = True

    def emit(self, stage, event, **detail):
        if self.callback:
            self.callback(stage, event, **detail)

    @contextmanager
    def stage(self, name):
        if not self.active:
            yield
            return
        self.emit(name, "started", path=[entry["name"] for entry in self.stack]+[name])
        started = self.clock()
        entry = dict(name=name, children=0.)
        self.stack.append(entry)
        event, error = "completed", None
        try:
            # Starting progress is flushed BEFORE synchronization, so a stuck
            # CUDA barrier is attributable even when the call never returns.
            self.synchronize()
            yield
            self.synchronize()
        except BaseException as exc:
            event, error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = max(0., (self.clock()-started)*1000)
            self.stack.pop()
            self.inclusive[name] += elapsed
            self.exclusive[name] += max(0., elapsed-entry["children"])
            self.counts[name] += 1
            if self.stack:
                self.stack[-1]["children"] += elapsed
            self.emit(name, event, elapsed_ms=elapsed, error=error,
                      path=[entry["name"] for entry in self.stack]+[name])

    def finish(self):
        self.active = False
        return self.report()

    def report(self):
        return dict(stage_ms=dict(self.exclusive), stage_total_ms=dict(self.inclusive),
                    stage_counts=dict(self.counts),
                    timing_method="CUDA-synchronized wall time; stage_ms is exclusive")


class TimedProxy:
    """Delegate to the same native object/model; time only specified methods."""
    def __init__(self, wrapped, timings, methods, children=None):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_timings", timings)
        object.__setattr__(self, "_methods", methods)
        object.__setattr__(self, "_children", children or {})

    def __getattr__(self, name):
        if name in self._children:
            return self._children[name]
        value = getattr(self._wrapped, name)
        stage = self._methods.get(name)
        if stage is None or not callable(value):
            return value

        @wraps(value)
        def call(*args, **kwargs):
            with self._timings.stage(stage):
                return value(*args, **kwargs)
        return call

    def __setattr__(self, name, value):
        setattr(self._wrapped, name, value)


class StageProgress:
    """Append-only evidence plus atomic latest progress; no GPU dependency."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = (self.folder/"stages.jsonl").open("a", buffering=1)
        self.context = {}

    def begin(self, packet):
        self.context = {key: packet[key] for key in
                        ("token", "command_id", "target", "frame_id")}

    def emit(self, stage, event, **detail):
        row = dict(self.context, stage=stage, event=event, at=time.time(), **detail)
        encoded = json.dumps(row)
        self.stream.write(encoded+"\n")
        self.stream.flush()
        temporary = self.folder/"progress.tmp"
        temporary.write_text(encoded+"\n")
        temporary.replace(self.folder/"progress.json")

    def close(self):
        self.stream.close()
