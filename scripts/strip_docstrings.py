import ast
from pathlib import Path

FILES = [
    Path('env/environment.py'),
    Path('env/grader.py'),
    Path('env/tasks.py'),
    Path('env/models.py'),
    Path('baseline/run.py'),
    Path('server/app.py'),
    Path('server/netra_environment.py'),
]

for path in FILES:
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    removals: list[tuple[int, int]] = []

    def mark_docstring(node: ast.AST) -> None:
        body = getattr(node, 'body', None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr):
            value = first.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                removals.append((first.lineno, first.end_lineno))

    mark_docstring(tree)
    for child in ast.walk(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark_docstring(child)

    lines = source.splitlines()
    remove_numbers = set()
    for start, end in removals:
        remove_numbers.update(range(start, end + 1))

    filtered = [line for index, line in enumerate(lines, start=1) if index not in remove_numbers]
    text = '\n'.join(filtered)
    if source.endswith('\n'):
        text += '\n'
    text = text.replace('# Reuse the OpenEnv app at the root level so reset/step/state stay compatible.\n', '')
    text = text.replace('\n\n\n', '\n\n')
    path.write_text(text, encoding='utf-8')
