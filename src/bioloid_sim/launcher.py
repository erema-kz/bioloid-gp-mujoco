from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tkinter as tk


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _python_executable() -> Path:
    executable = Path(sys.executable)
    pythonw = executable.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else executable


def launch_arena(root: tk.Tk) -> None:
    subprocess.Popen(
        [str(_python_executable()), "-m", "bioloid_sim.arena"],
        cwd=PROJECT_ROOT,
    )
    root.destroy()


def launch_training(root: tk.Tk) -> None:
    subprocess.Popen(
        ["cmd.exe", "/k", str(PROJECT_ROOT / "launch_visible_training.cmd")],
        cwd=PROJECT_ROOT,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    root.destroy()


def main() -> None:
    root = tk.Tk()
    root.title("Bioloid GP")
    root.geometry("520x390")
    root.resizable(False, False)
    root.configure(bg="#101923")

    panel = tk.Frame(root, bg="#101923", padx=32, pady=28)
    panel.pack(fill="both", expand=True)

    tk.Label(
        panel,
        text="BIOLOID GP",
        font=("Segoe UI Semibold", 11),
        fg="#65B8FF",
        bg="#101923",
    ).pack(anchor="w")
    tk.Label(
        panel,
        text="Центр управления роботом",
        font=("Segoe UI Semibold", 22),
        fg="#F3F7FA",
        bg="#101923",
    ).pack(anchor="w", pady=(4, 8))
    tk.Label(
        panel,
        text="Выбери режим запуска",
        font=("Segoe UI", 10),
        fg="#A9B8C6",
        bg="#101923",
    ).pack(anchor="w", pady=(0, 22))

    arena_button = tk.Button(
        panel,
        text="АРЕНА  •  несколько роботов",
        command=lambda: launch_arena(root),
        font=("Segoe UI Semibold", 12),
        fg="white",
        bg="#1668C7",
        activeforeground="white",
        activebackground="#1E78E3",
        relief="flat",
        cursor="hand2",
        pady=13,
    )
    arena_button.pack(fill="x")

    training_button = tk.Button(
        panel,
        text="ОБУЧЕНИЕ ИИ  •  подъём и ходьба",
        command=lambda: launch_training(root),
        font=("Segoe UI Semibold", 12),
        fg="#071B1B",
        bg="#43C6B7",
        activeforeground="#071B1B",
        activebackground="#58D7C9",
        relief="flat",
        cursor="hand2",
        pady=13,
    )
    training_button.pack(fill="x", pady=(12, 0))

    tk.Label(
        panel,
        text="Обучение: жизнь 7 сек • 35% стартов лёжа • 3 млн шагов\n"
        "Не запускай две тренировки одновременно.",
        font=("Segoe UI", 9),
        justify="left",
        fg="#D6A84D",
        bg="#101923",
    ).pack(anchor="w", pady=(18, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
