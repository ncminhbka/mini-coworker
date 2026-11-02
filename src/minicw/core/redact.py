"""Che khoá API và token trước khi ghi log ra đĩa.

Một khoá lọt vào bản ghi (lỗi 401 in lại header, model chép `env` ra, lệnh shell echo biến môi
trường) sẽ nằm trong messages.json, trajectory.json, summary.json và results.jsonl, những file
được chép đi, gửi cho giảng viên, đôi khi commit nhầm. Che ở lúc ghi, không phụ thuộc vào việc
mọi nơi phát sinh chuỗi đều nhớ che.

Chép từ OpenWorker `headless/runner.py:463-485` @ 59151e3 (MIT, andrewyng/openworker): cùng danh
sách hậu tố tên biến, cùng ngưỡng 8 ký tự và cùng hai mẫu khoá. Khác: tách thành module riêng,
đưa `re` lên đầu file, và có thêm `redact_json` cho dữ liệu đã là dict/list.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Biến môi trường có tên chứa một trong các đoạn này được coi là chứa bí mật.
_SECRET_ENV_MARKERS = ("_API_KEY", "_TOKEN", "_SECRET", "PASSWORD")
# Khoá chỉ có vài ký tự (ví dụ "1234") thì thay thế khắp văn bản sẽ phá hỏng nội dung vô tội, nên
# chỉ che giá trị từ 8 ký tự trở lên.
_MIN_SECRET_LEN = 8
_SK_KEY = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
_AWS_KEY = re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")


def _secret_values() -> list[str]:
    """Giá trị của các biến môi trường trông như thông tin xác thực, dài trước ngắn sau để giá
    trị dài không bị che dở bởi một giá trị ngắn nằm trong nó."""
    values = []
    for name, value in os.environ.items():
        upper = name.upper()
        if value and len(value.strip()) >= _MIN_SECRET_LEN and any(m in upper for m in _SECRET_ENV_MARKERS):
            values.append(value.strip())
    return sorted(set(values), key=len, reverse=True)


def redact(text: str) -> str:
    """Che giá trị của biến môi trường chứa bí mật và các dạng khoá phổ biến (sk-..., AKIA...)."""
    for value in _secret_values():
        text = text.replace(value, "[REDACTED]")
    # Các dạng khoá thường gặp, phòng trường hợp giá trị đến bằng đường khác ngoài môi trường.
    text = _SK_KEY.sub("[REDACTED]", text)
    return _AWS_KEY.sub("[REDACTED]", text)


def redact_json(data: Any, **dumps_kwargs: Any) -> str:
    """json.dumps rồi che. Che trên văn bản đã serialize để không phải đi qua từng khoá của dict."""
    return redact(json.dumps(data, ensure_ascii=False, default=str, **dumps_kwargs))
