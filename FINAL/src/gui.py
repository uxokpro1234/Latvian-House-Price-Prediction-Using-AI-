from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.model_training as mt



class TrainerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Real Estate Price Trainer")
        self.selected_files: list[str] = []

        self.select_btn = tk.Button(root, text="Select Training CSV Files", command=self.select_files)
        self.select_btn.pack(pady=6)

        self.train_btn = tk.Button(root, text="Train Model", command=self.start_training)
        self.train_btn.pack(pady=6)

        self.status_label = tk.Label(root, text="No training started yet.", anchor="w", justify="left")
        self.status_label.pack(pady=6, fill="x")

        self.metrics_box = tk.Text(root, height=12, width=70)
        self.metrics_box.pack(pady=6)
        self.metrics_box.insert("end", "Metrics will appear here after training.\n")

    def select_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select CSV Files", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if files:
            self.selected_files = list(files)
            self.status_label.config(text=f"Selected {len(files)} file(s).")
        else:
            self.status_label.config(text="No files selected.")

    def start_training(self) -> None:
        if not self.selected_files:
            messagebox.showwarning("No files", "Please select at least one CSV file.")
            return
        self.train_btn.config(state="disabled")
        self.status_label.config(text="Training in progress...")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        try:
            csv_paths = [Path(p) for p in self.selected_files]
            df = mt.load_datasets(csv_paths)
            result = mt.train_models(df)
            mt.save_artifact(result, result.model_path)

            self.root.after(0, self._on_success, result)

        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._on_error, exc)
        finally:
            self.root.after(0, self._on_cleanup)

    def _on_success(self, result: mt.TrainingResult) -> None:
        self._show_metrics(result)
        self.status_label.config(
            text=(
                f"Trained {result.best_model_name} on {result.training_rows} rows. "
                f"Saved to {result.model_path}."
            )
        )
        messagebox.showinfo(
            "Training finished",
            f"Best model: {result.best_model_name}\n"
            f"MAE: {result.metrics.get('mae', float('nan')):.2f}\n"
            f"RMSE: {result.metrics.get('rmse', float('nan')):.2f}\n"
            f"R2: {result.metrics.get('r2', float('nan')):.3f}",
        )

    def _on_error(self, exc: Exception) -> None:
        self.status_label.config(text=f"Error: {exc}")
        messagebox.showerror("Training failed", str(exc))

    def _on_cleanup(self) -> None:
        self.train_btn.config(state="normal")

    def _show_metrics(self, result: mt.TrainingResult) -> None:
        self.metrics_box.delete("1.0", "end")
        self.metrics_box.insert(
            "end",
            "\n".join(
                [
                    f"Best model: {result.best_model_name}",
                    f"Rows: {result.training_rows}",
                    f"MAE: {result.metrics.get('mae', float('nan')):.3f}",
                    f"RMSE: {result.metrics.get('rmse', float('nan')):.3f}",
                    f"R2: {result.metrics.get('r2', float('nan')):.3f}",
                    f"MAPE: {result.metrics.get('mape', float('nan')):.3f} %",
                    f"Annual growth rate: {result.annual_growth_rate:.4f}",
                ]
            ),
        )


def main() -> None:
    root = tk.Tk()
    TrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
