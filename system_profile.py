from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import os
import platform
import subprocess
from typing import Dict


@dataclass(frozen=True)
class SystemProfile:
    os_name: str
    architecture: str
    cpu_count: int
    total_memory_gb: float
    machine_tier: str
    is_apple_silicon: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _detect_total_memory_bytes() -> int:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)
        return 0

    if sys_platform() == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(output)
        except Exception:
            pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        return 0
    return 0


def sys_platform() -> str:
    return platform.system().lower()


def detect_machine_tier(total_memory_gb: float, override: str = "auto") -> str:
    normalized = (override or "auto").strip().lower()
    if normalized in {"compact", "balanced", "high-memory"}:
        return normalized
    if total_memory_gb >= 24:
        return "high-memory"
    if total_memory_gb >= 12:
        return "balanced"
    return "compact"


def detect_system_profile(machine_tier_override: str = "auto") -> SystemProfile:
    architecture = platform.machine().lower()
    os_name = sys_platform()
    total_memory_bytes = _detect_total_memory_bytes()
    total_memory_gb = round(total_memory_bytes / (1024 ** 3), 1) if total_memory_bytes else 0.0
    cpu_count = os.cpu_count() or 1
    is_apple_silicon = os_name == "darwin" and architecture in {"arm64", "aarch64"}
    machine_tier = detect_machine_tier(total_memory_gb, machine_tier_override)

    return SystemProfile(
        os_name=os_name,
        architecture=architecture,
        cpu_count=cpu_count,
        total_memory_gb=total_memory_gb,
        machine_tier=machine_tier,
        is_apple_silicon=is_apple_silicon,
    )
