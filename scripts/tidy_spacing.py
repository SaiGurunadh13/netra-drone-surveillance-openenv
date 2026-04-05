import re
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
    text = path.read_text(encoding='utf-8')
    text = text.lstrip('\n')
    text = re.sub(r'\):\n\n(\s+)', r'):\n\1', text)
    text = re.sub(r':\n\n(\s+)(?=return|if|for|while|try|import|from|[A-Za-z_])', r':\n\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    path.write_text(text, encoding='utf-8')
