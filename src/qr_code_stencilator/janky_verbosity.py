from rich import print as rprint
from rich.theme import Theme
rich_theme = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "error": "bold red"
})

VERBOSITY_LUT = {
    'ERROR'  : 1,
    'WARNING': 2,
    'DEBUG'  : 3,
    'SIMPLE' : 0,
    'SILENT' :-1,
}

def create_log_functions(verbosity):
    vbl = VERBOSITY_LUT[verbosity]
    print_error = lambda *a, **k: rprint(f"[error]{a[0]}[/error]{'\n' if len(a) >= 0 else ''}{'\n'.join(a[1:])}", **k) if vbl >= 1 else lambda *a, **k: None
    print_warning = lambda *a, **k: rprint(f"[warning]{a[0]}[/warning]{'\n' if len(a) > 1 else ''}{'\n'.join(a[1:])}", **k) if vbl >= 1 else lambda *a, **k: None
    print_debug = rprint if vbl >= 3 else lambda *a, **k: None
    print = rprint if vbl >= 0 else lambda *a, **k: None

    return print_error,print_warning,print_debug,print