---
trigger: always_on
---

Use venv in the workspace root to run Python code and tests.
> source venv/bin/activate

To run e2e tests, use '-o "addopts=-m e2e"' and '-s' with pytest