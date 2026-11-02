"""Which shell commands only read, and which files such a command would read.

Two questions, answered separately: `is_readonly_command` says what a command may DO (it must
be a single simple command from a short allowlist, without a flag that writes or runs another
program); `read_targets` names the files it would READ, so the caller can hold them to the
workspace. Without the second half `cat ~/.ssh/id_rsa` passes as "read-only".

Adapted from OpenWorker `coworker/readonly.py` @ 59151e3 (MIT, andrewyng/openworker):
`_flag_denied`, `_has_unquoted_shell_variable`, `_find_targets`, the operand rules in
`_stage_targets` and the idea of failing closed are taken from there. See docs/ for what mcw
simplifies (one command, no pipelines; a much shorter allowlist; PowerShell names added).
"""

from __future__ import annotations

import re
import shlex

# Lệnh chỉ-đọc được chạy luôn không cần hỏi. Gồm lệnh bash (ls, cat, grep...) lẫn PowerShell
# (get-childitem, get-content...) vì mcw chạy trên cả hai. So khớp theo từ đầu, chữ thường.
READONLY_COMMANDS = {
    "ls", "dir", "cat", "head", "tail", "wc", "pwd", "echo", "grep", "rg", "find", "type",
    "get-childitem", "get-content", "select-string", "get-location", "where", "which",
}
# Lệnh chỉ-đọc xét từ hai từ đầu, vì chỉ một số lệnh con là an toàn ("git status" được,
# "git push" thì không).
# Không có `pip`: `pip list --log ../x` ghi file, và OpenWorker cũng không cho pip.
READONLY_PREFIXES = (
    ("git", "status"), ("git", "log"), ("git", "diff"), ("git", "show"), ("git", "branch"),
    ("python", "--version"),
)

# Ký tự cho phép ghép nhiều lệnh, chuyển hướng hoặc thay thế lệnh: ; & | < > ` $( ) và xuống
# dòng (\n, \r tách lệnh như dấu ";"). Có một ký tự này thì không còn là "một lệnh đơn giản".
# Khác OpenWorker: họ cho phép pipe `|` nếu mọi giai đoạn đều qua kiểm tra, mcw thì không.
_SHELL_OPERATORS = re.compile(r"[;&|<>`\n\r]|\$\(")
# Ngoài dấu nháy, các ký tự này cũng biến một "lệnh đơn" thành thứ khác, vì lệnh chạy trong
# PowerShell (Windows) chứ không chỉ bash:
#   ( ) và @( ): đối số trong ngoặc được CHẠY như lệnh: `echo (New-Item x)` tạo file;
#   { }: khối lệnh (`where -InputObject 1 {ni x}` chạy khối), và brace expansion của bash
#        (`cat {..,x}/s` thành `cat ../s x/s`);
#   , : danh sách trong PowerShell: `Get-Content a,C:\secret` đọc cả hai file.
# Trong nháy thì vô hại (PowerShell không chạy "(x)", còn `$(` đã bị chặn ở trên).
_UNQUOTED_OPERATORS = re.compile(r"[(){},]")

# Cờ biến một lệnh trong danh sách thành lệnh ghi file, xoá file hoặc chạy chương trình khác.
# Giá trị là các cờ bị cấm; cách so khớp xem _flag_denied.
_FIND_BAD = ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls", "-fprintf")
_FLAG_BAD: dict[str, tuple[str, ...]] = {
    "find": _FIND_BAD,
    "rg": ("--pre", "--pre-glob", "--hostname-bin", "--pager"),
}
# git: --output ghi kết quả ra file. "git branch" chỉ liệt kê khi mọi đối số nằm trong tập
# này; đối số khác (tên nhánh, -D, -m...) tạo, xoá hoặc đổi tên nhánh.
_GIT_BRANCH_SAFE = {"-a", "-r", "-v", "-vv", "--all", "--remotes", "--list", "--show-current"}


def _has_unquoted_shell_variable(command: str) -> bool:
    """True nếu có `$` nằm ngoài nháy đơn. Biến được shell thay giá trị SAU khi mình kiểm tra,
    nên `cat $HOME/x` có thể đọc bất cứ đâu. Trong nháy đơn thì `$` là ký tự thường."""
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif char == "'" and quote != '"':
            quote = None if quote == "'" else "'"
        elif char == '"' and quote != "'":
            quote = None if quote == '"' else '"'
        elif char == "$" and quote != "'":
            return True
    return False


def _unquoted(command: str) -> str:
    """`command` với mọi đoạn trong nháy đơn hoặc nháy kép bị bỏ đi."""
    out, quote = [], None
    for char in command:
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        else:
            out.append(char)
    return "".join(out)


def _argv(command: str) -> list[str] | None:
    """Tách một lệnh đơn thành các từ, hoặc None nếu không phải lệnh đơn đủ đơn giản."""
    if _SHELL_OPERATORS.search(command) or _has_unquoted_shell_variable(command):
        return None
    if _UNQUOTED_OPERATORS.search(_unquoted(command)):
        return None
    # posix=False để giữ dấu "\" trong đường dẫn Windows (chế độ posix coi "\" là ký tự thoát
    # và làm hỏng đường dẫn). Hệ quả: dấu nháy còn dính trong từ, nên bóc ra ở dưới.
    try:
        raw = shlex.split(command, posix=False)
    except ValueError:  # nháy không đóng
        return None
    argv = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"" else t for t in raw]
    return argv or None


def _flag_denied(token: str, denied: tuple[str, ...]) -> bool:
    """True khi `token` là một cờ trong `denied`, hoặc là cờ đó kèm giá trị gắn liền.

    `sort --output=out`, `sort -oout` và `sort -o out` là cùng một cờ; so sánh nguyên chuỗi sẽ
    bỏ lọt hai dạng đầu. Cờ dài khớp `flag=` đúng (nên `--output-delimiter` không bị nhầm với
    `--output`); cờ ngắn 2 ký tự khớp mọi giá trị gắn liền, kể cả trong cụm cờ (`-ro`).
    Nhầm thừa chỉ tốn một lần hỏi, bỏ lọt mới nguy hiểm.
    """
    if not token.startswith("-") or token in ("-", "--"):
        return False
    for flag in denied:
        if token == flag:
            return True
        if flag.startswith("--"):
            if token.startswith(flag + "="):
                return True
        elif len(flag) == 2 and not token.startswith("--"):
            if flag[1] in token[1:]:
                return True
        elif token.startswith(flag):
            return True
    return False


def is_readonly_command(command: str) -> bool:
    """True khi `command` là MỘT lệnh đơn chỉ-đọc từ danh sách. Gặp gì nghi ngờ thì False."""
    argv = _argv(command)
    if argv is None:
        return False
    # Lệnh gọi bằng đường dẫn (/bin/ls, C:\tools\ls.exe) có thể là bất kỳ chương trình nào.
    if "/" in argv[0] or "\\" in argv[0]:
        return False
    head = argv[0].lower()
    args = [a.lower() for a in argv[1:]]
    if head in READONLY_COMMANDS:
        return not any(_flag_denied(a, _FLAG_BAD.get(head, ())) for a in args)
    for prefix in READONLY_PREFIXES:
        if tuple(a.lower() for a in argv[: len(prefix)]) != prefix:
            continue
        rest = args[len(prefix) - 1:]
        # --output ghi file; --no-index biến `git diff` thành so hai file tuỳ ý trên đĩa
        # (`git diff --no-index ../../secret x`); --ext-diff / --textconv chạy chương trình.
        # --no-index cũng lọt ở OpenWorker `readonly.py:337-347`.
        if head == "git" and any(a.startswith(("--output", "--no-index", "--ext-diff", "--textconv"))
                                 for a in rest):
            return False
        if prefix == ("git", "branch") and any(a not in _GIT_BRANCH_SAFE for a in rest):
            return False
        return True
    return False


# -- file mà lệnh sẽ đọc ----------------------------------------------------------------
# is_readonly_command nói lệnh được LÀM gì, không nói nó ĐỌC gì. Các hàm dưới trả về các đối số
# có vẻ là đường dẫn, để nơi gọi kiểm tra chúng nằm trong workspace. Không thể làm đúng cho
# shell nói chung; ở đây làm được vì lệnh đã bị thu hẹp vào vài lệnh quen thuộc.

# Đối số không phải đường dẫn (chuỗi, tên lệnh).
_NO_PATH_OPERANDS = {"echo", "pwd", "get-location", "where", "which"}
# Đối số đầu tiên không phải cờ là mẫu tìm kiếm, không phải file; các đối số sau mới là file.
# select-string cố ý KHÔNG nằm đây: PowerShell dùng tham số có tên (-Path f -Pattern x), nên
# không đoán được cái nào là mẫu; coi mọi đối số là đường dẫn thì chỉ thừa, không thiếu.
_PATTERN_FIRST = {"grep", "rg"}
# Cờ mà GIÁ TRỊ của nó là đường dẫn.
_PATH_VALUE_FLAGS = {"-f", "--file", "--exclude-from", "--include-from"}
# `head -n 5`, `grep -m 2`: số thuần là đối số của một cờ, không phải file.
_NUMERIC = re.compile(r"^[0-9]+([,:.-][0-9]+)*[a-zA-Z]?$")
# Cờ toàn cục của find đứng TRƯỚC điểm bắt đầu tìm, nên không được kết thúc danh sách đường dẫn.
_FIND_GLOBAL_NO_VALUE = frozenset({"-H", "-L", "-P", "-E", "-X", "-d", "-s", "-x"})


def _find_targets(args: list[str]) -> list[str]:
    """Điểm bắt đầu của `find`, bỏ qua cờ toàn cục. `find -L /etc` đọc /etc giống `find /etc`."""
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in _FIND_GLOBAL_NO_VALUE or (tok.startswith("-O") and tok != "-O"):
            i += 1
            continue
        if tok == "-D" and i + 1 < len(args):
            i += 2  # tuỳ chọn gỡ lỗi, không phải đường dẫn
            continue
        if tok == "-f" and i + 1 < len(args):
            out.append(args[i + 1])  # BSD: -f là điểm bắt đầu
            i += 2
            continue
        break
    # Điểm bắt đầu kết thúc ở biểu thức đầu tiên (token bắt đầu bằng "-"). Không có điểm nào
    # thì find mặc định là ".", luôn nằm trong workspace.
    while i < len(args) and not args[i].startswith("-"):
        out.append(args[i])
        i += 1
    return out


def read_targets(command: str) -> list[str]:
    """Các đối số của `command` có thể là file bị đọc. Chỉ có nghĩa với lệnh đã qua
    `is_readonly_command`. Thà liệt kê thừa (tốn một lần hỏi) còn hơn thiếu (đọc ngoài tầm kiểm soát).

    Giới hạn đã biết: đường dẫn đi qua một cờ không có trong bảng thì không được trả về
    (ví dụ `rg --files ../x`), cũng như bản gốc.
    """
    argv = _argv(command)
    if not argv:
        return []
    head, args = argv[0].lower(), argv[1:]
    if head in _NO_PATH_OPERANDS or head in ("python", "pip"):
        return []
    if head == "git":
        # Chỉ `--` mở ra pathspec; mcw không cho cờ toàn cục như `-C <dir>` đứng trước lệnh con.
        for j, tok in enumerate(args):
            if tok == "--":
                return [t for t in args[j + 1:] if not t.startswith("-")]
        return []
    if head == "find":
        return _find_targets(args)

    out: list[str] = []
    skip_next = False
    pattern_next = False
    seen_operand = False
    for tok in args:
        if pattern_next:  # giá trị của -e / --regexp là mẫu, không phải file
            pattern_next = False
            continue
        if skip_next:
            # `-f` là file mẫu của grep nhưng là số thứ tự trường của cut; kiểm tra số để phân biệt.
            if not _NUMERIC.match(tok):
                out.append(tok)
            seen_operand = True  # mẫu đã đến từ cờ, nên đối số tiếp theo là file
            skip_next = False
            continue
        if tok.startswith("-"):
            name, sep, value = tok.partition("=") if tok.startswith("--") else tok.partition(":")
            if sep and value:
                # Giá trị gắn liền: `--file=x` (bash) hoặc `-Path:C:\x` (PowerShell). Với
                # PowerShell không biết tham số nào nhận đường dẫn, nên coi mọi giá trị như vậy
                # là đường dẫn; nhầm thừa chỉ tốn một lần hỏi.
                if not tok.startswith("--") or name in _PATH_VALUE_FLAGS:
                    out.append(value)
                if name == "--regexp":
                    seen_operand = True
                continue
            if head in _PATTERN_FIRST:
                if tok in ("-e", "--regexp"):
                    pattern_next, seen_operand = True, True
                    continue
                if tok.startswith("-e") and not tok.startswith("--"):
                    # `-e.`: mẫu gắn liền với cờ. Bản gốc không xét dạng này, nên file đầu
                    # tiên sau nó bị bỏ qua như là mẫu (OpenWorker `readonly.py:370-372`).
                    seen_operand = True
                    continue
            if tok in _PATH_VALUE_FLAGS:
                skip_next = True
            continue
        if head in _PATTERN_FIRST and not seen_operand:
            seen_operand = True  # đây là mẫu tìm kiếm, bỏ qua
            continue
        seen_operand = True
        if not _NUMERIC.match(tok):
            out.append(tok)
    return out
