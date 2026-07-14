"""训练日志记录模块。

将所有终端输出（print、tqdm 进度条、警告等）同时写入日志文件，
方便事后查看和分析训练过程。

用法:
    from src.utils.logger import setup_logger

    log_path = setup_logger("logs/train_solar.log")
    print("这条信息会同时出现在终端和日志文件里")
    # ... 训练代码 ...
"""

import os
import sys
import datetime
from typing import Optional, TextIO


class Tee:
    """将输出同时写入终端和文件（类似 Unix tee 命令）。

    终端写入保持原样（tqdm 用 \\r 原地刷新进度条）。
    文件写入模拟终端的 \\r 行为：\\r 回退到当前行首，
    后续内容覆盖缓冲行，只有遇到 \\n 才真正写入文件。
    这样日志文件中每个 tqdm 进度条只占一行，干净整洁。
    """

    def __init__(self, file_path: str, stream: TextIO):
        self.file = open(file_path, "a", encoding="utf-8", buffering=1)
        self.stream = stream
        self._closed = False
        self._file_buf = ""  # 当前待写入文件的缓冲行

    def write(self, message: str):
        """写入消息：终端原样输出，文件处理 \\r 为行覆盖。"""
        if self._closed:
            return

        # 终端保持原样（tqdm \\r 正常工作）
        self.stream.write(message)

        # 文件：模拟终端的 \\r / \\n 行为
        for ch in message:
            if ch == '\r':
                # 回到当前行首（最后一个 \\n 之后）
                nl = self._file_buf.rfind('\n')
                self._file_buf = self._file_buf[:nl + 1] if nl >= 0 else ''
            elif ch == '\n':
                self._file_buf += ch
                self.file.write(self._file_buf)
                self._file_buf = ''
            else:
                self._file_buf += ch

    def flush(self):
        """刷新缓冲区。"""
        if self._closed:
            return
        self.stream.flush()
        self.file.flush()

    def close(self):
        """关闭文件句柄（先 flush 残留缓冲行）。"""
        if not self._closed:
            if self._file_buf:
                self.file.write(self._file_buf + '\n')
                self._file_buf = ''
            self.file.close()
            self._closed = True

    def fileno(self):
        """返回原始流的文件描述符（兼容需要 fileno 的库）。"""
        return self.stream.fileno()


class TeeLogger:
    """管理 stdout/stderr 到日志文件的重定向。

    支持 with 语句上下文管理器用法，确保在异常退出时也能正确恢复。
    """

    def __init__(self, log_path: str):
        """初始化日志记录器。

        Args:
            log_path: 日志文件路径。父目录会在不存在时自动创建。
        """
        self.log_path = log_path
        self._stdout_tee: Optional[Tee] = None
        self._stderr_tee: Optional[Tee] = None
        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None

    def start(self):
        """开始记录日志。替换 sys.stdout 和 sys.stderr。"""
        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 保存原始流
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # 创建 Tee 并替换
        self._stdout_tee = Tee(self.log_path, self._original_stdout)
        self._stderr_tee = Tee(self.log_path, self._original_stderr)

        sys.stdout = self._stdout_tee  # type: ignore[assignment]
        sys.stderr = self._stderr_tee  # type: ignore[assignment]

        # 写入日志头
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"{'='*60}\n日志开始: {timestamp}\n{'='*60}\n\n"
        self._original_stdout.write(header)
        self._stdout_tee.file.write(header)

    def stop(self):
        """停止记录日志，恢复原始 sys.stdout 和 sys.stderr。"""
        # 恢复原始流
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout  # type: ignore[assignment]
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr  # type: ignore[assignment]

        # 关闭 Tee 文件
        if self._stdout_tee is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"\n{'='*60}\n日志结束: {timestamp}\n{'='*60}\n"
            self._stdout_tee.file.write(footer)
            self._stdout_tee.close()
        if self._stderr_tee is not None:
            self._stderr_tee.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # 不吞异常


def setup_logger(log_path: Optional[str] = None, prefix: str = "train") -> str:
    """便捷函数：根据时间戳生成日志路径并启动日志记录。

    Args:
        log_path: 自定义日志文件路径。若为 None，在 logs/ 目录下自动生成
                  带时间戳的文件名，如 logs/train_2026-07-14_14-35-52.log。
        prefix: 日志文件名前缀（仅在 log_path 为 None 时使用）。

    Returns:
        日志文件的绝对路径。
    """
    if log_path is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(logs_dir, f"{prefix}_{timestamp}.log")

    # 实例化并启动
    logger = TeeLogger(log_path)
    logger.start()

    # 返回路径后将 logger 实例存储在模块全局中以备后用
    _active_logger = logger

    return os.path.abspath(log_path)
