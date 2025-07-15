import pkgutil
import importlib
from typing import List
from langchain.tools import BaseTool


def load_all_tools() -> List[BaseTool]:
    discovered_tools = []
    package = importlib.import_module('tools')

    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        if module_name == __name__:
            continue
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BaseTool):
                print(f"✅ Discovered tool: '{attr.name}' from {module_name}")
                discovered_tools.append(attr)
    if not discovered_tools:
        raise ImportError("No tools found.")
    return discovered_tools
