# vestige-client (Python SDK)

Zero-config Python client and 1-line drop-in adapters for the **Vestige Causal Proof Engine** (`#832`).

## Install
```bash
pip install vestige-client
```

## Quickstart
```python
from vestige_client import VestigeClient
from vestige_client.adapters.langchain import VestigeStore
from vestige_client.adapters.crewai import VestigeMemory
from vestige_client.adapters.llamaindex import VestigeRetriever
from vestige_client.adapters.autogen import VestigeCausalMemory

client = VestigeClient()
receipt = client.remember("Auth tokens expire in 15m", node_type="decision", tags=["auth"])
found = client.recall(handle="auth")
```
