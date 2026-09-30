#!/usr/bin/env python3
"""CPU test: the tool_choice:"none" patcher matches the image's glm47 parser
anchors, and the patched ``adjust_request`` masks ``<tool_call>`` only for
none+tools chat requests (client ``bad_words`` preserved, other requests,
the Responses API shape and the ``super()`` chain untouched)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _d in (HERE, ROOT / "overlay"):
    if (_d / "patch_tool_choice_none.py").is_file():
        sys.path.insert(0, str(_d))
        break
from patch_tool_choice_none import MARK, OLD, apply_text  # noqa: E402

# Dependency-free harness carrying the exact image anchor.
MINIMAL = (
    'TOOL_CALL_START = "<tool_call>"\n'
    "\n"
    "class ChatCompletionRequest:\n"
    "    def __init__(self, tool_choice, tools, bad_words=None):\n"
    "        self.tool_choice = tool_choice\n"
    "        self.tools = tools\n"
    "        self.bad_words = [] if bad_words is None else bad_words\n"
    "        self.skip_special_tokens = True\n"
    "\n"
    "class ResponsesRequest:\n"
    "    def __init__(self, tool_choice, tools):\n"
    "        self.tool_choice = tool_choice\n"
    "        self.tools = tools\n"
    "        self.skip_special_tokens = True\n"
    "\n"
    "class ParserEngine:\n"
    "    def adjust_request(self, request):\n"
    "        request.skip_special_tokens = False\n"
    "        return request\n"
    "    def is_reasoning_end(self, input_ids):\n"
    "        return True\n"
    "\n"
    "class Glm47MoeParser(ParserEngine):\n"
    "    thinking_enabled = False\n"
    "\n"
    f"{OLD}"
)

TOOLS = [{"type": "function", "function": {"name": "lookup_fact"}}]


def _patched_parser() -> dict[str, object]:
    out, status = apply_text(MINIMAL)
    assert status == "applied", status
    ns: dict[str, object] = {}
    exec(compile(out, "patched_glm47_moe_fixture.py", "exec"), ns)
    return ns


def test_apply_then_skip() -> None:
    out, status = apply_text(MINIMAL)
    assert status == "applied", status
    assert MARK in out and out.count(MARK) == 1
    assert "request.bad_words.append(TOOL_CALL_START)" in out
    out2, status2 = apply_text(out)
    assert status2 == "skipped", status2
    assert out2 == out


def test_missing_anchor_fails_closed() -> None:
    _, status = apply_text("class Glm47MoeParser:\n    pass\n")
    assert status.startswith("missing:"), status
    drifted = MINIMAL.replace('"<tool_call>"', '"<call>"')
    _, status = apply_text(drifted)
    assert status.startswith("missing:"), status


def test_none_with_tools_masks_opener() -> None:
    ns = _patched_parser()
    parser = ns["Glm47MoeParser"]()
    req = parser.adjust_request(ns["ChatCompletionRequest"]("none", TOOLS))
    assert req.bad_words == ["<tool_call>"], req.bad_words
    assert req.skip_special_tokens is False


def test_client_bad_words_preserved() -> None:
    ns = _patched_parser()
    parser = ns["Glm47MoeParser"]()
    req = parser.adjust_request(ns["ChatCompletionRequest"]("none", TOOLS, ["foo"]))
    assert req.bad_words == ["foo", "<tool_call>"], req.bad_words


def test_other_requests_untouched() -> None:
    ns = _patched_parser()
    parser = ns["Glm47MoeParser"]()
    make = ns["ChatCompletionRequest"]
    for tool_choice, tools in (("auto", TOOLS), ("required", TOOLS),
                               ("none", None), ("none", [])):
        req = parser.adjust_request(make(tool_choice, tools))
        assert req.bad_words == [], (tool_choice, tools, req.bad_words)
        assert req.skip_special_tokens is False
    resp = parser.adjust_request(ns["ResponsesRequest"]("none", TOOLS))
    assert not hasattr(resp, "bad_words")
    assert resp.skip_special_tokens is False


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'TOOLCHOICE_PATCH_HOST="${TOOLCHOICE_PATCH_HOST:-' in launcher
    order = launcher[launcher.index("GLM53_OVERLAY_ORDER=(") : launcher.index(")", launcher.index("GLM53_OVERLAY_ORDER=("))]
    assert "\n    patch_tool_choice_none.py\n" in order
    assert 'emit_overlay_block >> "$HEAD_SCRIPT"' in launcher
    assert 'emit_overlay_block >> "$WORKER_SCRIPT"' in launcher
    assert (
        "-v '/tmp/patch_tool_choice_none.py:"
        "/opt/glm53/patch_tool_choice_none.py:ro'" in launcher
    )
    assert (
        '-v "$TOOLCHOICE_PATCH_HOST:'
        '/opt/glm53/patch_tool_choice_none.py:ro"' in launcher
    )
    assert 'scp -q -o BatchMode=yes "$TOOLCHOICE_PATCH_HOST"' in launcher
    assert "COPY overlay/patch_tool_choice_none.py" in image
    assert "RUN python3 /opt/glm53/patch_tool_choice_none.py" in image
    assert "python3 /opt/glm53/test_tool_choice_none.py" in image


if __name__ == "__main__":
    test_apply_then_skip()
    test_missing_anchor_fails_closed()
    test_none_with_tools_masks_opener()
    test_client_bad_words_preserved()
    test_other_requests_untouched()
    test_recipe_wiring_if_present()
    print("test_tool_choice_none: ok")
