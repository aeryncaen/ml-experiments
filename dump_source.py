import inspect
from heuristic_secrets.models import backbone

output_path = '/tmp/backbone_recovered.py'

with open(output_path, 'w') as f:
    f.write(inspect.getsource(backbone))

print(f"Source written to {output_path}")
