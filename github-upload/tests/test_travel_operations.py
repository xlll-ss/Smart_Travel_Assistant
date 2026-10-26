import logging

import pytest

from travel_assistant import (
    LOGGER,
    TravelPreferenceStore,
    TravelPreferences,
    UserKnowledgeBaseManager,
    configure_logging,
    parse_auth_config,
    resolve_user_identity,
    user_facing_error,
)


def make_preferences(city: str) -> TravelPreferences:
    return TravelPreferences(
        city=city,
        mood="想安静",
        activity_preference="室内为主",
        budget="经济",
        duration_hours=4,
        companions="独自出行",
        departure="市中心",
    )


def test_preferences_are_isolated_by_user(tmp_path):
    store = TravelPreferenceStore(tmp_path / "preferences.db")
    store.save("user-a", make_preferences("杭州"))
    store.save("user-b", make_preferences("上海"))

    assert store.load("user-a").city == "杭州"
    assert store.load("user-b").city == "上海"
    assert store.load("unknown-user") is None


def test_private_knowledge_bases_are_isolated(tmp_path):
    manager = UserKnowledgeBaseManager(tmp_path / "users")
    try:
        user_a = manager.for_user("user-a")
        user_b = manager.for_user("user-b")
        user_a.add_guide("杭州攻略", "杭州博物馆需要提前核实预约信息。")

        assert len(user_a.list_documents()) == 1
        assert user_b.list_documents() == []
        assert user_a.rag_tool.database_path != user_b.rag_tool.database_path
    finally:
        manager.close_all()


def test_pasted_chinese_guide_can_be_retrieved(tmp_path):
    knowledge_base = UserKnowledgeBaseManager(tmp_path / "users").for_user(
        "guide-search-user"
    )
    try:
        result = knowledge_base.add_guide(
            "我的专属攻略",
            "我今天只想去人少的地方，比如某某小众书店。",
        )
        report = knowledge_base.search("根据我的攻略，推荐去哪里？")

        assert result["success"] is True
        assert report.snippets
        assert "人少的地方" in report.snippets[0].content
    finally:
        knowledge_base.close()


def test_request_identity_uses_authentication_or_session():
    authenticated = resolve_user_identity("alice", "session-one")
    same_authenticated = resolve_user_identity("alice", "session-two")
    anonymous_a = resolve_user_identity(None, "session-one")
    anonymous_b = resolve_user_identity(None, "session-two")

    assert authenticated == same_authenticated
    assert anonymous_a != anonymous_b
    assert authenticated != anonymous_a


def test_auth_configuration_parser():
    assert parse_auth_config(None) is None
    assert parse_auth_config("alice:secret,bob:other") == [
        ("alice", "secret"),
        ("bob", "other"),
    ]
    with pytest.raises(ValueError, match="user:password"):
        parse_auth_config("missing-password")


def test_internal_error_is_hidden_but_input_error_is_visible():
    internal = user_facing_error(RuntimeError("database password leaked"), "abc123")
    validation = user_facing_error(ValueError("城市不能为空"), "def456")

    assert "database password" not in internal
    assert "abc123" in internal
    assert "城市不能为空" in validation
    assert "def456" in validation


def test_rotating_log_file_is_created(tmp_path):
    log_path = configure_logging(tmp_path)
    LOGGER.info("event=test_log_write")
    for handler in LOGGER.handlers:
        handler.flush()

    assert log_path.exists()
    assert "event=test_log_write" in log_path.read_text(encoding="utf-8")
    assert any(isinstance(handler, logging.Handler) for handler in LOGGER.handlers)
