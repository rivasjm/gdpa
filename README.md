# Gradient Descent Priority Assignment

## Example

To install the dependencies, it is recommended to create virtual environment, and install the dependencies there.

In Windows (CMD):
```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

In Windows (Powershell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In Linux/Mac:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With the dependencies installed, the provided example can be executed with this command:

```
python main.py
```

If it executed correctly, this should be the output:
```
HOPA:  schedulable=False, invslack=0.0902742815917806
GDPA:  schedulable=True, invslack=-0.03343164541780137
```
