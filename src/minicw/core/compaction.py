"""Compaction of long histories.

When the outbound history nears the context window, the older part is replaced, in what is
SENT to the model, by (a) an LLM-written summary and (b) state extracted mechanically from
the span: the user's messages and the files/commands touched. The stored transcript is never
modified, so `boundary` is an index into an append-only list and stays valid.

Cuts fall before any non-`tool` message. That keeps an assistant message with `tool_calls`
together with its tool results (a provider rejects an orphaned tool reply), and unlike
"cut at a user turn" it still works when one task is a single huge tool loop. The first user
message (the task statement) is always kept whole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .provider import ProviderClient

# -- Hằng số ----------------------------------------------------------------------------------
# Ngưỡng kích hoạt nén: dùng 80% cửa sổ ngữ cảnh. Vượt mức này thì engine bắt đầu nén.
DEFAULT_THRESHOLD_PCT = 0.8
# Cửa sổ ngữ cảnh mặc định khi người dùng không truyền --context-window (128 nghìn token).
DEFAULT_CONTEXT_WINDOW = 128_000
# Khi nén xong, phần đuôi giữ nguyên văn được giới hạn ở 25% ngưỡng (xem pick_boundary).
KEEP_RECENT_FRACTION = 0.25
# 16k, not 4k: every backend mcw uses is a reasoning model, and the thinking spends from the
# same budget (OpenWorker raised theirs for the same reason, `compaction.py:34`, OPE-189).
# Trần số token đầu ra của lời gọi tóm tắt. Đặt lớn vì model suy nghĩ dùng chung ngân sách này.
SUMMARY_MAX_TOKENS = 16_000
# Bản tóm tắt ngắn hơn 200 ký tự thì coi là hỏng (summary_problem).
SUMMARY_MIN_CHARS = 200
# Ngân sách token cho các tin nhắn của người dùng được giữ lại sau khi nén.
USER_BUDGET_TOKENS = 2_000
# Tin nhắn đầu (đề bài) được giữ nguyên tối đa 4000 token, quá thì cắt bớt.
PIN_MAX_TOKENS = 4_000  # the task statement is kept whole up to this, then clipped
# Mỗi tool result trong phần được tóm tắt chỉ đưa vào prompt 400 ký tự đầu.
TOOL_RESULT_CLIP = 400
# Toàn bộ phần văn bản đưa cho model tóm tắt không vượt 200 nghìn ký tự (cắt lấy phần cuối).
SPAN_BUDGET_CHARS = 200_000
# When no summary can be produced, the boundary moves past only this share of the remaining
# messages, not all the way to the keep point: losing a little at a time beats losing 75% at
# once (OpenWorker `compaction.py:67,621-666`, `trim_state`).
# Khi không có bản tóm tắt, chỉ bỏ 10% số tin nhắn còn lại (xem _trim_boundary).
TRIM_FRACTION = 0.10
# Giới hạn số file và số lệnh được liệt kê trong trạng thái làm việc, để nó không phình ra.
MAX_FILES_LISTED = 20
MAX_COMMANDS_LISTED = 10
# Bảy mục bắt buộc của bản tóm tắt. Thứ tự này cũng là thứ tự model phải viết.
SECTIONS = (
    "Primary request", "Key facts", "Files and artifacts", "Errors and fixes",
    "Pending tasks", "Current work", "Next step",
)
# Prompt hệ thống gửi cho model tóm tắt: yêu cầu đúng bảy mục và cấm nó tiếp tục phiên.
SUMMARIZER_PROMPT = (
    "You are compacting the earlier part of a working session so the work can continue "
    "without it. Write a summary with exactly these sections, each as a '## ' heading: "
    + ", ".join(SECTIONS)
    + ". Be specific: names, paths, numbers, decisions and what remains. Do not continue "
    "the session and do not call tools."
)
# Restated AFTER the transcript. The transcript usually ends mid-task, on a tool result, and a
# model that read that last tends to continue the session instead of summarizing it (13 of 16
# short summaries in OpenWorker's trial, `compaction.py:553-567`, OPE-189).
# Câu nhắc đặt SAU transcript, vì model dễ tiếp tục việc đang làm nếu chữ cuối cùng nó đọc là kết quả tool.
SUMMARY_TAIL = (
    "--- END OF TRANSCRIPT ---\n\nThe transcript above is a session you are summarizing, not a "
    "conversation you are part of. Do not continue it or take its next action. Write the "
    f"summary now, beginning with the line '## {SECTIONS[0]}'."
)


# -- Ước lượng token ----------------------------------------------------------------------------
# Không dùng tokenizer thật mà ước lượng bằng số ký tự chia 4: nhanh và không phụ thuộc model.
def _tokens(message: dict[str, Any]) -> int:
    # ensure_ascii=False: with the default, every Vietnamese letter becomes a 6-character
    # \\uXXXX escape and the estimate runs ~2.5x high ("lương tháng": 11 characters, 28 escaped).
    # Chuyển tin nhắn thành JSON rồi đếm ký tự. ensure_ascii=False giữ nguyên chữ có dấu,
    # nếu không thì mỗi chữ có dấu thành chuỗi \uXXXX dài 6 ký tự và ước lượng cao gấp ~2.5 lần.
    return len(json.dumps(message, ensure_ascii=False, default=str)) // 4


# Tổng ước lượng token của một danh sách tin nhắn.
def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_tokens(m) for m in messages)


# Ngưỡng nén tính bằng token: cửa sổ * 80%. Nếu có cap_tokens thì lấy giá trị nhỏ hơn.
def trigger_tokens(context_window: int = DEFAULT_CONTEXT_WINDOW, threshold_pct: float = DEFAULT_THRESHOLD_PCT,
                   cap_tokens: Optional[int] = None) -> int:
    trigger = int(context_window * threshold_pct)
    # `if cap_tokens` là False khi cap_tokens là None hoặc 0, lúc đó không giới hạn thêm.
    return min(trigger, cap_tokens) if cap_tokens else trigger


# Trạng thái nén, lưu trong engine.compaction_state. Mỗi lần nén tạo ra một object mới.
@dataclass
class CompactionState:
    boundary: int  # messages[boundary:] are sent verbatim; everything before is replaced
    summary: str  # bản tóm tắt do LLM viết ("" nếu chưa có)
    working_state: str  # file đã ghi/đọc và lệnh đã chạy, trích bằng code
    user_messages: list[str]  # các tin nhắn của người dùng nằm trong phần đã nén
    trimmed: bool = False  # True when no summary could be produced

    # Saved with the session so a reopened session keeps its compacted view (reference
    # `compaction.py:164-191`, `as_dict` / `from_dict`).
    def as_dict(self) -> dict[str, Any]:
        return {"boundary": self.boundary, "summary": self.summary, "working_state": self.working_state,
                "user_messages": list(self.user_messages), "trimmed": self.trimmed}

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["CompactionState"]:
        """None when there is nothing usable (never compacted, or a broken record)."""
        if not isinstance(data, dict) or not isinstance(data.get("boundary"), int):
            return None
        return cls(boundary=data["boundary"], summary=str(data.get("summary") or ""),
                   working_state=str(data.get("working_state") or ""),
                   user_messages=[str(m) for m in data.get("user_messages") or []],
                   trimmed=bool(data.get("trimmed")))

    def valid_for(self, messages: list[dict[str, Any]]) -> bool:
        """A saved state is only trusted if its boundary points inside the history and at a
        message that is not a tool result (a cut must never separate a call from its result)."""
        return 1 <= self.boundary < len(messages) and messages[self.boundary].get("role") != "tool"


# Chọn điểm cắt: tìm vị trí cắt sớm nhất sao cho phần đuôi (từ điểm cắt đến hết) có token không vượt keep_tokens.
def pick_boundary(messages: list[dict[str, Any]], *, start: int, keep_tokens: int) -> Optional[int]:
    """Earliest valid cut whose tail is at most `keep_tokens`; when even the newest step is
    bigger than that, the newest valid cut (at least the latest step stays verbatim). None
    when the span before the cut would be empty. Walks back from the end accumulating tokens.
    "At most", not "at least": a tail kept above the budget can leave the prompt over the
    trigger right after compacting, and compaction then fires again every iteration."""
    total = 0  # tổng token của phần đuôi, cộng dần khi đi ngược từ cuối
    cut = latest = None  # cut: điểm cắt tốt nhất tìm được; latest: điểm cắt hợp lệ gần cuối nhất
    # Đi ngược từ tin cuối về đến start (không gồm start, vì start là điểm đầu của phần được nén).
    for i in range(len(messages) - 1, start, -1):
        total += _tokens(messages[i])
        # Không cắt trước tin `tool`: phải giữ tool_calls đi cùng kết quả của nó.
        if messages[i].get("role") == "tool":
            continue
        # Điểm cắt hợp lệ đầu tiên gặp khi đi ngược là điểm gần cuối nhất.
        if latest is None:
            latest = i
        # Phần đuôi đã vượt ngân sách thì dừng: cut giữ điểm cắt hợp lệ cuối cùng còn trong ngân sách.
        if total > keep_tokens:
            break
        cut = i
    # Nếu không có điểm nào vừa ngân sách, lấy điểm gần cuối nhất (giữ ít nhất bước mới nhất).
    cut = cut if cut is not None else latest
    # Không có điểm cắt, hoặc phần trước điểm cắt rỗng (chỉ có start): không đáng nén.
    if cut is None or cut <= start + 1:  # nothing worth compacting
        return None
    return cut


# Lấy chữ từ nội dung tin nhắn. Nội dung có thể là chuỗi, danh sách các phần (kiểu multimodal) hoặc None.
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Chỉ lấy các phần có khoá "text", nối bằng dấu cách.
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


# Trích trạng thái làm việc bằng code (không nhờ LLM): file đã ghi, đã đọc và lệnh đã chạy trong đoạn.
def extract_working_state(span: list[dict[str, Any]]) -> str:
    """Files written/read and commands run in the span, with their outcome, from the tool
    calls themselves. Nothing here depends on the summarizer behaving."""
    # Map từ tool_call_id tới nội dung kết quả, để biết một lệnh gọi có lỗi hay không.
    results = {m.get("tool_call_id"): _text_of(m.get("content")) for m in span if m.get("role") == "tool"}
    wrote, read, ran = [], [], []  # ba danh sách: file ghi, file đọc, lệnh chạy
    for m in span:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            # Tham số là chuỗi JSON; nếu hỏng thì bỏ qua lệnh gọi này.
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                continue
            name = fn.get("name")
            # Lệnh bị coi là lỗi nếu kết quả của nó có khoá "error".
            failed = '"error"' in results.get(tc.get("id"), "")
            if name in ("write_file", "replace_in_file"):
                wrote.append(f"{args.get('path')}{' (failed)' if failed else ''}")
            elif name == "read_file":
                read.append(str(args.get("path")))
            elif name == "run_shell":
                # Chỉ lấy 120 ký tự đầu của lệnh để dòng không quá dài.
                ran.append(f"{str(args.get('command'))[:120]}{' (failed)' if failed else ''}")
    # Bounded however long the session runs: the most recent distinct files, the last
    # commands (limits from OpenWorker `compaction.py:317-341`).
    # Giới hạn độ dài: chỉ giữ file gần nhất và vài lệnh cuối để kích thước không tăng mãi.
    parts = []
    if wrote:
        parts.append("Files written: " + ", ".join(_recent_distinct(wrote, MAX_FILES_LISTED)))
    if read:
        parts.append("Files read: " + ", ".join(_recent_distinct(read, MAX_FILES_LISTED)))
    if ran:
        parts.append("Commands run: " + "; ".join(ran[-MAX_COMMANDS_LISTED:]))
    return "\n".join(parts)


# Lấy tối đa `limit` mục khác nhau, ưu tiên mục xuất hiện gần đây nhất, giữ thứ tự xuất hiện lần cuối.
def _recent_distinct(items: list[str], limit: int) -> list[str]:
    """The last `limit` distinct items, in the order they last appeared."""
    # dict dùng như tập có thứ tự: đi từ cuối lên, mục đã thấy thì bỏ qua.
    seen: dict[str, None] = {}
    for item in reversed(items):
        if item not in seen:
            seen[item] = None
            if len(seen) >= limit:
                break
    # Đảo lại để thứ tự trở về đúng chiều thời gian.
    return list(reversed(seen))


# Cắt chuỗi về khoảng `tokens` token (một token ~ 4 ký tự), thêm "…" nếu bị cắt.
def _clip_to_tokens(text: str, tokens: int) -> str:
    limit = max(1, tokens) * 4  # đổi token thành số ký tự, tối thiểu 1 token
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Lấy nội dung của mọi tin nhắn người dùng trong đoạn.
def user_texts(span: list[dict[str, Any]]) -> list[str]:
    return [_text_of(m.get("content")) for m in span if m.get("role") == "user"]


# Giữ các tin nhắn người dùng mới nhất trong ngân sách token; tin mới nhất luôn được giữ dù vượt ngân sách.
def fit_user_messages(texts: list[str], budget_tokens: int = USER_BUDGET_TOKENS) -> list[str]:
    """Newest first until the budget is spent, returned in order. The newest one is kept even
    when it alone is over the budget, clipped: it is the last thing the user actually said
    (OpenWorker `compaction.py:413-417`)."""
    kept: list[str] = []
    used = 0  # số token đã dùng trong ngân sách
    # reversed: duyệt từ tin mới nhất về trước.
    for text in reversed(texts):
        cost = len(text) // 4 + 1  # ước lượng số token của tin này (+1 để không bằng 0)
        # Đã có ít nhất một tin và tin này làm vượt ngân sách thì dừng. Tin đầu tiên (mới nhất) thì luôn được giữ.
        if kept and used + cost > budget_tokens:
            break
        kept.append(_clip_to_tokens(text, budget_tokens))
        used += cost
    # Đảo lại để thứ tự là thứ tự thời gian gốc.
    return list(reversed(kept))


# Chuyển đoạn tin nhắn thành một khối văn bản để đưa cho model tóm tắt.
def _render_span(span: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in span:
        role = m.get("role")
        if role == "tool":
            # Kết quả tool chỉ lấy 400 ký tự đầu.
            lines.append(f"[tool result] {_text_of(m.get('content'))[:TOOL_RESULT_CLIP]}")
        elif role == "assistant":
            # Một tin assistant có thể gọi nhiều tool: ghi từng lệnh gọi, kèm 300 ký tự đầu của tham số.
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                lines.append(f"[assistant called {fn.get('name')}] {str(fn.get('arguments'))[:300]}")
            if _text_of(m.get("content")):
                lines.append(f"[assistant] {_text_of(m.get('content'))}")
        else:
            lines.append(f"[{role}] {_text_of(m.get('content'))}")
    # Nếu quá dài thì giữ phần cuối (phần gần nhất thường quan trọng hơn).
    return "\n".join(lines)[-SPAN_BUDGET_CHARS:]


# Kiểm tra chất lượng bản tóm tắt. Trả về lý do nếu không đạt, None nếu đạt.
def summary_problem(text: Optional[str]) -> Optional[str]:
    """Why a summary is unacceptable, or None. The summary is the only memory of the span,
    so 'not empty' is too weak a bar."""
    # Quá ngắn (hoặc không có) thì không đủ thông tin để thay cho phần đã nén.
    if not text or len(text.strip()) < SUMMARY_MIN_CHARS:
        return "too short"
    # Phải có tiêu đề mục đầu tiên ("Primary request"), không phân biệt hoa thường.
    if SECTIONS[0].lower() not in text.lower():
        return "missing sections"
    return None


# Gọi model để tóm tắt một đoạn. Trả về bản tóm tắt, hoặc None nếu thất bại sau khi thử lại.
def summarize_span(
    provider: ProviderClient, model: str, span: list[dict[str, Any]], previous_summary: str = "",
    on_usage: Optional[Callable[[Any], None]] = None,
) -> Optional[str]:
    body = _render_span(span)
    # Nếu đã có tóm tắt của các lần nén trước, đặt nó lên đầu để không mất thông tin cũ.
    if previous_summary:
        body = f"Summary of the session before this span:\n{previous_summary}\n\n---\n\n{body}"
    # Prompt hệ thống (yêu cầu tóm tắt) + tin người dùng (nội dung đoạn, kèm câu nhắc cuối).
    messages = [{"role": "system", "content": SUMMARIZER_PROMPT},
                {"role": "user", "content": f"{body}\n\n{SUMMARY_TAIL}"}]
    # Thử tối đa 2 lần: lỗi gọi API hoặc tóm tắt không đạt đều được thử lại một lần.
    for _ in range(2):  # one retry, for a failed call as much as for a bad summary
        try:
            # tools=None: không cho model gọi tool khi đang tóm tắt.
            turn = provider.complete(model=model, messages=messages, tools=None, max_tokens=SUMMARY_MAX_TOKENS)
        except Exception:  # noqa: BLE001 - compaction must never kill the run
            # Lỗi gọi API không làm chết run: bỏ qua và thử lại.
            continue
        # Lời gọi tóm tắt cũng tốn tiền, báo lại để engine tính vào trần chi phí.
        if on_usage and turn.usage is not None:
            on_usage(turn.usage)  # the summary is a paid call too; the cost caps must see it
        # A summary cut off at the limit has lost its last sections (Pending tasks, Next step).
        # Bản tóm tắt bị cắt do giới hạn đầu ra thì mất các mục cuối, nên coi là hỏng.
        if turn.finish_reason != "length" and summary_problem(turn.text) is None:
            return turn.text.strip()
    # Hết lượt thử mà vẫn không có bản tóm tắt tốt.
    return None


# Tạo khối văn bản thay thế phần đã nén, sẽ được chèn vào bản gửi đi cho model.
def compacted_block(state: CompactionState) -> str:
    # Mở đầu: nói rõ cho model biết phần trước đã bị nén.
    parts = ["[Earlier part of this session was compacted. What follows is all that is kept of it.]"]
    if state.summary:
        parts.append(state.summary)
    # Nếu không có tóm tắt, báo cho model biết những tin mới nhất đã bị bỏ và nên đọc lại file.
    if state.trimmed:
        parts.append("(No summary could be produced for the latest turns, so they were dropped. "
                     "Re-read files and re-run commands if you need earlier results.)")
    if state.working_state:
        parts.append("Working state:\n" + state.working_state)
    if state.user_messages:
        parts.append("The user's messages in the compacted span:\n" + "\n---\n".join(state.user_messages))
    # Ghép các phần bằng dòng trống.
    return "\n\n".join(parts)


# Tạo bản gửi đi cho model: tin đề bài, khối nén, rồi phần đuôi giữ nguyên văn. Không đụng vào transcript gốc.
def apply_to_outbound(messages: list[dict[str, Any]], state: Optional[CompactionState]) -> list[dict[str, Any]]:
    """The view sent to the model: first user message, the compacted block, then the tail."""
    # Chưa nén lần nào thì gửi nguyên transcript (một bản sao để không ai sửa list gốc).
    if state is None:
        return list(messages)
    head = messages[0]  # the task statement, never compacted away (clipped if huge)
    text = _text_of(head.get("content"))
    # Đề bài quá dài thì cắt bớt trong bản gửi đi. {**head, ...} tạo dict mới, không sửa dict gốc.
    if len(text) > PIN_MAX_TOKENS * 4:
        head = {**head, "content": _clip_to_tokens(text, PIN_MAX_TOKENS)}
    block = {"role": "user", "content": compacted_block(state)}
    # Bản gửi đi = đề bài + khối nén + messages từ boundary trở đi (phần chưa bị nén).
    return [head, block, *messages[state.boundary:]]


# Điểm cắt khi không có bản tóm tắt: bỏ khoảng 10% số tin còn lại, nhưng không cắt ngay trước một tin `tool`.
def _trim_boundary(messages: list[dict[str, Any]], start: int) -> Optional[int]:
    """The no-summary fallback's cut: ~TRIM_FRACTION of the remaining messages, landing on a
    message that is not a tool result."""
    remaining = len(messages) - start  # số tin còn lại từ start
    # Quá ít tin thì không cắt.
    if remaining <= 2:
        return None
    # Bắt đầu tìm từ vị trí start + 10%, tiến về cuối cho đến khi gặp tin không phải tool.
    for i in range(start + max(1, int(remaining * TRIM_FRACTION)), len(messages)):
        if messages[i].get("role") != "tool":
            return i if i > start else None
    return None


# Hàm chính của nén: tạo CompactionState mới, hoặc trả về None nếu không có gì để cắt.
def compact(
    provider: ProviderClient, model: str, messages: list[dict[str, Any]],
    previous: Optional[CompactionState], *, trigger: int,
    on_usage: Optional[Callable[[Any], None]] = None, force: bool = False,
) -> Optional[CompactionState]:
    """Compute a new state, or None when there is nothing to cut. `force` (the provider said
    the prompt is too long) trims even when no cut fits the keep budget."""
    # Lần đầu bắt đầu từ 1 (bỏ qua đề bài ở vị trí 0). Các lần sau bắt đầu từ điểm cắt cũ.
    start = previous.boundary if previous else 1  # index 0 is the pinned task
    # Tìm điểm cắt mới, với phần đuôi không vượt 25% ngưỡng.
    boundary = pick_boundary(messages, start=start, keep_tokens=int(trigger * KEEP_RECENT_FRACTION))
    # Bản tóm tắt cũ sẽ được đưa vào prompt tóm tắt lần này.
    previous_summary = previous.summary if previous else ""
    if boundary is None:
        # Không tìm được điểm cắt. Nếu bị ép (force) thì cắt theo kiểu trim, không cần tóm tắt.
        boundary = _trim_boundary(messages, start) if force else None
        if boundary is None:
            return None
        summary = None
    else:
        # Có điểm cắt: tóm tắt đoạn [start, boundary). Có thể trả về None nếu tóm tắt thất bại.
        summary = summarize_span(provider, model, messages[start:boundary], previous_summary, on_usage)
    # Tóm tắt thất bại mà không bị ép: chỉ cắt thêm một ít, giữ bản tóm tắt cũ. Lần nén sau sẽ thử lại.
    if summary is None and not force:
        # Keep the earlier summary and drop only a little more; the next compaction tries again.
        boundary = _trim_boundary(messages, start)
        if boundary is None:
            return None
    # Các tin người dùng từ lần nén trước được giữ lại và cộng thêm tin trong đoạn mới.
    carried = previous.user_messages if previous else []
    return CompactionState(
        boundary=boundary,
        # Có tóm tắt mới thì dùng; không có thì giữ tóm tắt cũ.
        summary=summary if summary is not None else previous_summary,
        # Rebuilt from the whole compacted history, so it stays complete and bounded.
        # Dựng lại từ toàn bộ phần đã nén (từ 1 đến boundary), nên luôn đầy đủ và có giới hạn.
        working_state=extract_working_state(messages[1:boundary]),
        user_messages=fit_user_messages(carried + user_texts(messages[start:boundary])),
        # trimmed = True khi không có tóm tắt mới, để compacted_block báo cho model biết.
        trimmed=summary is None,
    )


# Nhận diện lỗi "tràn ngữ cảnh" từ provider bằng cách tìm các cụm từ đặc trưng trong thông báo lỗi.
def is_context_overflow(exc: BaseException) -> bool:
    # Gộp thông báo lỗi và phần body (nếu có) rồi đổi sang chữ thường để so khớp.
    text = (str(exc) + " " + str(getattr(exc, "body", ""))).lower()
    return any(k in text for k in (
        "context length", "context_length", "maximum context", "too many tokens", "prompt is too long",
        "input is too long", "exceeds the maximum number of tokens", "input length and `max_tokens` exceed",
    ))
