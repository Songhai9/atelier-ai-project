"""Execute selected learning notebooks in a fresh project Python kernel."""

from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parents[1]
names = sys.argv[1:] or [p.name for p in sorted((root / "notebooks").glob("*.ipynb"))]
allowed = {p.name for p in (root / "notebooks").glob("*.ipynb")}
for name in names:
    if name not in allowed:
        raise ValueError(f"Unknown notebook: {name}")
    path = root / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]

    def started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"{name}: executing cell {cell_index}", flush=True)

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=3600,
        resources={"metadata": {"path": str(root / "notebooks")}},
        on_cell_start=started,
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"{name}: saved executed notebook", flush=True)
