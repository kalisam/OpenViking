from types import SimpleNamespace

import pytest

from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.retrieve.types import FindResult


def test_operation_telemetry_summary_includes_memory_extract_breakdown():
    from openviking.telemetry.operation import OperationTelemetry

    telemetry = OperationTelemetry(operation="session.commit", enabled=True)
    telemetry.set("memory.extracted", 5)
    telemetry.set("memory.extract.total.duration_ms", 842.3)
    telemetry.set("memory.extract.candidates.total", 7)
    telemetry.set("memory.extract.candidates.standard", 5)
    telemetry.set("memory.extract.candidates.tool_skill", 2)
    telemetry.set("memory.extract.created", 3)
    telemetry.set("memory.extract.merged", 1)
    telemetry.set("memory.extract.deleted", 0)
    telemetry.set("memory.extract.skipped", 3)
    telemetry.set("memory.extract.stage.prepare_inputs.duration_ms", 8.4)
    telemetry.set("memory.extract.stage.llm_extract.duration_ms", 410.2)
    telemetry.set("memory.extract.stage.normalize_candidates.duration_ms", 6.7)
    telemetry.set("memory.extract.stage.tool_skill_stats.duration_ms", 1.9)
    telemetry.set("memory.extract.stage.profile_create.duration_ms", 12.5)
    telemetry.set("memory.extract.stage.tool_skill_merge.duration_ms", 43.0)
    telemetry.set("memory.extract.stage.dedup.duration_ms", 215.6)
    telemetry.set("memory.extract.stage.create_memory.duration_ms", 56.1)
    telemetry.set("memory.extract.stage.merge_existing.duration_ms", 22.7)
    telemetry.set("memory.extract.stage.delete_existing.duration_ms", 0.0)
    telemetry.set("memory.extract.stage.create_relations.duration_ms", 18.2)
    telemetry.set("memory.extract.stage.flush_semantic.duration_ms", 9.0)

    summary = telemetry.finish().summary

    assert summary["memory"]["extracted"] == 5
    assert summary["memory"]["extract"] == {
        "duration_ms": 842.3,
        "candidates": {
            "total": 7,
            "standard": 5,
            "tool_skill": 2,
        },
        "actions": {
            "created": 3,
            "merged": 1,
            "skipped": 3,
        },
        "stages": {
            "prepare_inputs_ms": 8.4,
            "llm_extract_ms": 410.2,
            "normalize_candidates_ms": 6.7,
            "tool_skill_stats_ms": 1.9,
            "profile_create_ms": 12.5,
            "tool_skill_merge_ms": 43.0,
            "dedup_ms": 215.6,
            "create_memory_ms": 56.1,
            "merge_existing_ms": 22.7,
            "create_relations_ms": 18.2,
            "flush_semantic_ms": 9.0,
        },
    }


def test_operation_telemetry_measure_accumulates_duration(monkeypatch):
    from openviking.telemetry.operation import OperationTelemetry

    perf_values = iter([10.0, 10.1, 10.3, 10.5, 10.8, 11.0])
    monkeypatch.setattr(
        "openviking.telemetry.operation.time.perf_counter", lambda: next(perf_values)
    )

    telemetry = OperationTelemetry(operation="session.commit", enabled=True)
    with telemetry.measure("memory.extract.stage.dedup"):
        pass
    with telemetry.measure("memory.extract.stage.dedup"):
        pass

    summary = telemetry.finish().summary
    assert summary["duration_ms"] == 1000.0
    assert summary["memory"]["extract"]["stages"]["dedup_ms"] == 500.0


def test_operation_telemetry_summary_includes_resource_breakdown():
    from openviking.telemetry.operation import OperationTelemetry

    telemetry = OperationTelemetry(operation="resources.add_resource", enabled=True)
    telemetry.set("resource.request.duration_ms", 152.3)
    telemetry.set("resource.process.duration_ms", 101.7)
    telemetry.set("resource.parse.duration_ms", 38.1)
    telemetry.set("resource.parse.warnings_count", 1)
    telemetry.set("resource.finalize.duration_ms", 22.4)
    telemetry.set("resource.summarize.duration_ms", 31.8)
    telemetry.set("resource.wait.duration_ms", 46.9)
    telemetry.set("resource.watch.duration_ms", 0.8)
    telemetry.set("resource.flags.wait", True)
    telemetry.set("resource.flags.build_index", True)
    telemetry.set("resource.flags.summarize", False)
    telemetry.set("resource.flags.watch_enabled", False)

    summary = telemetry.finish().summary

    assert summary["resource"] == {
        "request": {"duration_ms": 152.3},
        "process": {
            "duration_ms": 101.7,
            "parse": {"duration_ms": 38.1, "warnings_count": 1},
            "finalize": {"duration_ms": 22.4},
            "summarize": {"duration_ms": 31.8},
        },
        "wait": {"duration_ms": 46.9},
        "watch": {"duration_ms": 0.8},
        "flags": {
            "wait": True,
            "build_index": True,
            "summarize": False,
            "watch_enabled": False,
        },
    }


def test_operation_telemetry_summary_omits_zero_valued_fields():
    from openviking.telemetry.operation import OperationTelemetry

    telemetry = OperationTelemetry(operation="resources.add_resource", enabled=True)
    telemetry.set("queue.semantic.processed", 0)
    telemetry.set("queue.semantic.error_count", 0)
    telemetry.set("queue.embedding.processed", 4)
    telemetry.set("queue.embedding.error_count", 0)
    telemetry.set("semantic_nodes.total", 9)
    telemetry.set("semantic_nodes.done", 8)
    telemetry.set("semantic_nodes.pending", 1)
    telemetry.set("semantic_nodes.running", 0)
    telemetry.set("resource.process.duration_ms", 12.3)
    telemetry.set("resource.parse.duration_ms", 0.0)
    telemetry.set("resource.parse.warnings_count", 0)
    telemetry.set("resource.flags.wait", False)
    telemetry.set("resource.flags.build_index", True)

    summary = telemetry.finish().summary

    assert "tokens" not in summary
    assert "semantic" not in summary["queue"]
    assert summary["queue"]["embedding"] == {"processed": 4}
    assert "running" not in summary["semantic_nodes"]
    assert summary["resource"] == {
        "process": {"duration_ms": 12.3},
        "flags": {"wait": False, "build_index": True, "summarize": False, "watch_enabled": False},
    }


def test_operation_telemetry_summary_includes_search_stage_durations():
    from openviking.telemetry.operation import OperationTelemetry

    telemetry = OperationTelemetry(operation="search.search", enabled=True)
    telemetry.set("search.vlm.duration_ms", 12.4)
    telemetry.set("search.embedding.duration_ms", 3.1)
    telemetry.set("search.vector_db.duration_ms", 18.9)
    telemetry.set("search.rerank.duration_ms", 5.6)

    summary = telemetry.finish().summary

    assert summary["search"] == {
        "vlm": {"duration_ms": 12.4},
        "embedding": {"duration_ms": 3.1},
        "vector_db": {"duration_ms": 18.9},
        "rerank": {"duration_ms": 5.6},
    }


def test_operation_telemetry_summary_includes_session_stage_durations():
    from openviking.telemetry.operation import OperationTelemetry

    telemetry = OperationTelemetry(operation="session.commit", enabled=True)
    telemetry.set("session.load.duration_ms", 12.0)
    telemetry.set("session.load.messages.duration_ms", 4.0)
    telemetry.set("session.load.history.duration_ms", 3.0)
    telemetry.set("session.load.meta.duration_ms", 2.0)
    telemetry.set("session.directory_init.duration_ms", 40.0)
    telemetry.set("session.directory_init.user_dirs.duration_ms", 18.0)
    telemetry.set("session.directory_init.agent_dirs.duration_ms", 22.0)
    telemetry.set("session.directory_init.agfs.duration_ms", 14.0)
    telemetry.set("session.directory_init.vector_db.duration_ms", 7.0)
    telemetry.set("session.directory_init.embedding.duration_ms", 5.0)
    telemetry.set("session.ensure_exists.duration_ms", 6.0)
    telemetry.set("session.message.append.duration_ms", 11.0)
    telemetry.set("session.message.meta.duration_ms", 9.0)
    telemetry.set("session.commit.lock_wait.duration_ms", 21.0)
    telemetry.set("session.commit.lock_hold.duration_ms", 33.0)
    telemetry.set("session.commit.write_live.duration_ms", 12.0)
    telemetry.set("session.commit.write_archive.duration_ms", 17.0)
    telemetry.set("session.commit.save_meta.duration_ms", 8.0)
    telemetry.set("session.commit.create_task.duration_ms", 1.0)
    telemetry.set("session.phase2.wait_previous.duration_ms", 2.0)
    telemetry.set("session.phase2.redo_log.duration_ms", 1.5)
    telemetry.set("session.phase2.summary.duration_ms", 120.0)
    telemetry.set("session.phase2.vlm.duration_ms", 95.0)
    telemetry.set("session.phase2.memory_extract.duration_ms", 260.0)
    telemetry.set("session.phase2.relations.duration_ms", 6.0)
    telemetry.set("session.phase2.active_count.duration_ms", 4.0)
    telemetry.set("session.phase2.finalize_meta.duration_ms", 7.0)
    telemetry.set("session.phase2.done_write.duration_ms", 3.0)
    telemetry.set("session.phase2.embedding.duration_ms", 54.0)
    telemetry.set("session.phase2.vector_db.duration_ms", 31.0)

    summary = telemetry.finish().summary

    assert summary["session"] == {
        "load_ms": 12.0,
        "load_messages_ms": 4.0,
        "load_history_ms": 3.0,
        "load_meta_ms": 2.0,
        "directory_init_ms": 40.0,
        "directory_init_user_dirs_ms": 18.0,
        "directory_init_agent_dirs_ms": 22.0,
        "directory_init_agfs_ms": 14.0,
        "directory_init_vector_db_ms": 7.0,
        "directory_init_embedding_ms": 5.0,
        "ensure_exists_ms": 6.0,
        "message_append_ms": 11.0,
        "message_meta_ms": 9.0,
        "commit_lock_wait_ms": 21.0,
        "commit_lock_hold_ms": 33.0,
        "commit_write_live_ms": 12.0,
        "commit_write_archive_ms": 17.0,
        "commit_save_meta_ms": 8.0,
        "commit_create_task_ms": 1.0,
        "phase2_wait_previous_ms": 2.0,
        "phase2_redo_log_ms": 1.5,
        "phase2_summary_ms": 120.0,
        "phase2_vlm_ms": 95.0,
        "phase2_memory_extract_ms": 260.0,
        "phase2_relations_ms": 6.0,
        "phase2_active_count_ms": 4.0,
        "phase2_finalize_meta_ms": 7.0,
        "phase2_done_write_ms": 3.0,
        "phase2_embedding_ms": 54.0,
        "phase2_vector_db_ms": 31.0,
    }


@pytest.mark.asyncio
async def test_run_with_telemetry_returns_usage_and_payload():
    from openviking.telemetry.execution import run_with_telemetry

    async def _run():
        return {"status": "ok"}

    execution = await run_with_telemetry(
        operation="search.find",
        telemetry=True,
        fn=_run,
    )

    assert execution.result == {"status": "ok"}
    assert execution.telemetry is not None
    assert execution.telemetry["summary"]["operation"] == "search.find"


@pytest.mark.asyncio
async def test_run_with_telemetry_raises_invalid_argument_for_bad_request():
    from openviking.telemetry.execution import run_with_telemetry

    async def _run():
        return {"status": "ok"}

    with pytest.raises(InvalidArgumentError, match="Unsupported telemetry options: invalid"):
        await run_with_telemetry(
            operation="search.find",
            telemetry={"invalid": True},
            fn=_run,
        )


@pytest.mark.asyncio
async def test_run_with_telemetry_rejects_events_selection():
    from openviking.telemetry.execution import run_with_telemetry

    async def _run():
        return {"status": "ok"}

    with pytest.raises(InvalidArgumentError, match="Unsupported telemetry options: events"):
        await run_with_telemetry(
            operation="search.find",
            telemetry={"summary": True, "events": False},
            fn=_run,
        )


def test_attach_telemetry_payload_adds_telemetry_to_dict_result():
    from openviking.telemetry.execution import attach_telemetry_payload

    result = attach_telemetry_payload(
        {"root_uri": "viking://resources/demo"},
        {
            "id": "1234567890abcdef1234567890abcdef",
            "summary": {"operation": "resources.add_resource"},
        },
    )

    assert result["telemetry"]["summary"]["operation"] == "resources.add_resource"


def test_attach_telemetry_payload_does_not_mutate_object_result():
    from openviking.telemetry.execution import attach_telemetry_payload

    result = SimpleNamespace(total=1)

    attached = attach_telemetry_payload(
        result,
        {"id": "1234567890abcdef1234567890abcdef", "summary": {"operation": "search.find"}},
    )

    assert attached is result
    assert not hasattr(result, "telemetry")


def test_find_result_ignores_usage_and_telemetry_payload_fields():
    result = FindResult.from_dict(
        {
            "memories": [],
            "resources": [],
            "skills": [],
            "telemetry": {
                "id": "1234567890abcdef1234567890abcdef",
                "summary": {"operation": "search.find"},
            },
        }
    )

    assert not hasattr(result, "telemetry")
    assert result.to_dict() == {
        "memories": [],
        "resources": [],
        "skills": [],
        "total": 0,
    }
