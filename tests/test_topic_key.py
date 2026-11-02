"""TopicKey (core/topic_key.py): the launcher, cwd, mode and model a topic resolves to.

Covers the precedence of each field (the rules of TurnRunner._plan), the empty-model rule, LauncherError when no launcher
resolves, and agreement with the plan of a real TurnRunner run on the same session row (test_plan_and_key_agree).
"""

import unittest

from tests.fakes.backend import FakeBackend
from tests.support import TempHome
from tests.turn_support import CHAT, THREAD, TOPIC, TurnHarness
from tg_bridge.config import Config, LauncherError
from tg_bridge.core.topic_key import TopicKey, topic_key, topic_launcher
from tg_bridge.store.sessions import SessionRecord

HOME = "/srv/tg-home"
OTHER = "OTHER"  # a topic with no entry in the config: the default topic (none here) and the stored values apply


def config(**data) -> Config:
    base = {
        "launchers": {"fake": "fake-cmd", "alt": "alt-cmd", "tlauncher": "topic-cmd"},
        "topics": {TOPIC: {"launcher": "topic-cmd", "cwd": "/srv/topic-cwd"}},
        "default_mode": "acceptEdits",
    }
    base.update(data)
    return Config(base, home=HOME, env={})


def row(**changes) -> SessionRecord:
    values = dict(thread_id=THREAD, name=TOPIC, uuid="", launcher="", cwd="", created="", title="", mode="",
                  model="", prev_uuid="", last_active="", launcher_pick="", backend="")
    values.update(changes)
    return SessionRecord(**values)


class TopicKeyTests(unittest.TestCase):
    def test_launcher_precedence_is_pick_then_topic_then_stored(self) -> None:
        cfg = config()
        cases = [
            ("pick", row(launcher_pick="alt", launcher="fake-cmd"), TOPIC, ("alt", "alt-cmd")),
            ("topic when no pick", row(launcher="fake-cmd"), TOPIC, ("tlauncher", "topic-cmd")),
            ("topic when the pick left the config", row(launcher_pick="gone", launcher="fake-cmd"), TOPIC,
             ("tlauncher", "topic-cmd")),
            ("stored when no pick and no topic entry", row(launcher="fake-cmd"), OTHER, ("fake", "fake-cmd")),
        ]
        for label, record, topic, expected in cases:
            with self.subTest(label):
                key = topic_key(cfg, record, topic)
                self.assertEqual((key.launcher_name, key.launcher_command), expected)
                self.assertEqual(topic_launcher(cfg, record, topic).command, expected[1])

    def test_cwd_precedence_is_row_then_topic_then_home(self) -> None:
        cfg = config()
        self.assertEqual(topic_key(cfg, row(cwd="/srv/row", launcher="fake-cmd"), TOPIC).cwd, "/srv/row")
        self.assertEqual(topic_key(cfg, row(launcher="fake-cmd"), TOPIC).cwd, "/srv/topic-cwd")
        self.assertEqual(topic_key(cfg, row(launcher="fake-cmd"), OTHER).cwd, HOME)

    def test_mode_precedence_is_row_then_default_mode(self) -> None:
        cfg = config()
        self.assertEqual(topic_key(cfg, row(mode="plan", launcher="fake-cmd"), TOPIC).mode, "plan")
        self.assertEqual(topic_key(cfg, row(launcher="fake-cmd"), TOPIC).mode, "acceptEdits")

    def test_empty_model_is_none(self) -> None:
        cfg = config()
        self.assertIsNone(topic_key(cfg, row(model="", launcher="fake-cmd"), TOPIC).model)
        self.assertEqual(topic_key(cfg, row(model="sonnet", launcher="fake-cmd"), TOPIC).model, "sonnet")

    def test_key_holds_every_field(self) -> None:
        key = topic_key(config(), row(cwd="/srv/row", mode="plan", model="opus", launcher_pick="alt",
                                      launcher="fake-cmd"), TOPIC)
        self.assertEqual(key, TopicKey(launcher_name="alt", launcher_command="alt-cmd", cwd="/srv/row",
                                       mode="plan", model="opus"))

    def test_launcher_error_propagates(self) -> None:
        cfg = config(launchers={}, topics={})
        with self.assertRaises(LauncherError):
            topic_launcher(cfg, row(), OTHER)
        with self.assertRaises(LauncherError):
            topic_key(cfg, row(), OTHER)


class PlanAgreesWithKeyTests(unittest.TestCase):
    """The plan TurnRunner builds for a row and the key of the same row are equal (one source of the precedence)."""

    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.h = TurnHarness(self.home.home)
        self.addCleanup(self.h.db.close)
        self.h.sessions.ensure(THREAD, TOPIC, "fake-cmd", "", "")  # an empty cwd: the topic cwd applies
        self.h.set_config(topics={TOPIC: {"launcher": "topic-cmd", "cwd": "/srv/topic-cwd"}}, default_mode="acceptEdits")
        self.runner = self.h.runner(self.h.registry_for(FakeBackend()))

    def assert_agree(self) -> TopicKey:
        plan = self.runner._plan(THREAD, CHAT, TOPIC, "hi", "tester", ())
        key = topic_key(self.h.config, self.h.sessions.get(THREAD), TOPIC)
        self.assertEqual(TopicKey(launcher_name=plan.launcher.name, launcher_command=plan.launcher.command,
                                  cwd=plan.cwd, mode=plan.mode, model=plan.model), key)
        return key

    def test_plan_and_key_agree(self) -> None:
        key = self.assert_agree()  # the fresh row: the topic's launcher and cwd, the configured default mode
        self.assertEqual(key, TopicKey("tlauncher", "topic-cmd", "/srv/topic-cwd", "acceptEdits", None))
        self.h.sessions.set_launcher_pick(THREAD, "alt")
        self.h.sessions.set_cwd(THREAD, "/srv/elsewhere")
        self.h.sessions.set_mode(THREAD, "default")
        self.h.sessions.set_model(THREAD, "sonnet")
        key = self.assert_agree()
        self.assertEqual(key, TopicKey("alt", "alt-cmd", "/srv/elsewhere", "default", "sonnet"))


if __name__ == "__main__":
    unittest.main()
