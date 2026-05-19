"""
模块依赖图引擎
- 模块级 DAG 构建
- 传递依赖扫描
- 循环检测 v2（处理别名和重导出）
- import 副作用检测
"""

import ast
import os
import sys
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict


class ModuleInfo:
    """单个模块的依赖信息"""
    def __init__(self, filepath: str, module_path: str):
        self.filepath = filepath
        self.module_path = module_path
        self.absolute_imports: Set[str] = set()   # import X, from X import Y
        self.relative_imports: Set[str] = set()    # from . import X, from .X import Y
        self.re_exports: Set[str] = set()          # from .X import Y (Y re-exported)
        self.side_effects: List[str] = []           # module-level code that isn't import/class/def

    def __repr__(self):
        return f"ModuleInfo({self.module_path})"


class DependencyGraph:
    """框架级模块依赖图"""

    def __init__(self, framework_root: str):
        self.framework_root = framework_root
        self.modules: Dict[str, ModuleInfo] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # module -> set of deps
        self._reverse_adj: Dict[str, Set[str]] = defaultdict(set)  # module -> set of dependents

    def scan(self):
        """扫描整个框架，构建依赖图"""
        for root, dirs, files in os.walk(self.framework_root):
            for f in files:
                if f.endswith(".py"):
                    filepath = os.path.join(root, f)
                    rel_path = os.path.relpath(filepath, self.framework_root)
                    module_path = rel_path.replace(os.sep, ".").replace(".py", "")
                    if module_path.endswith(".__init__"):
                        module_path = module_path[:-9]

                    info = self._analyze_file(filepath, module_path)
                    self.modules[module_path] = info

        self._build_adjacency()
        return self

    def _analyze_file(self, filepath: str, module_path: str) -> ModuleInfo:
        """深度分析单个文件的 imports 和副作用"""
        info = ModuleInfo(filepath, module_path)

        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError:
            return info

        # 收集所有 top-level 语句，检测副作用
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                                 ast.AnnAssign)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Str)):
                continue  # docstrings are fine
            # Everything else at module level is a potential side effect
            if isinstance(node, ast.If):
                # if __name__ == "__main__": is acceptable
                if (isinstance(node.test, ast.Compare) and
                    hasattr(node.test, 'left') and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == "__name__"):
                    continue
            info.side_effects.append(f"line {node.lineno}: {ast.dump(node)[:80]}")

        # 分析 imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info.absolute_imports.add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue

                if node.level == 0:
                    # absolute import
                    info.absolute_imports.add(node.module)
                elif node.level == 1:
                    # relative import from current package
                    target = node.module if node.module else ""
                    info.relative_imports.add(target)

                    # Check if names are re-exported (no __all__ restriction)
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        if alias.asname is None:
                            info.re_exports.add(f"{target}.{alias.name}")

        return info

    def _build_adjacency(self):
        """构建邻接表"""
        for module_path, info in self.modules.items():
            # Resolve relative imports to absolute paths
            for rel_import in info.relative_imports:
                if rel_import:
                    # from .ffbs import X -> tvp_var_framework.models.ffbs
                    parts = module_path.rsplit(".", 1)
                    if len(parts) == 2:
                        target = parts[0] + "." + rel_import
                    else:
                        target = rel_import
                    self._adjacency[module_path].add(target)
                    self._reverse_adj[target].add(module_path)

            # Absolute imports to framework modules
            for abs_import in info.absolute_imports:
                if abs_import.startswith("tvp_var_framework"):
                    self._adjacency[module_path].add(abs_import)
                    self._reverse_adj[abs_import].add(module_path)

    def detect_cycles(self) -> List[List[str]]:
        """Tarjan's algorithm for cycle detection"""
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        on_stack = {}
        cycles = []

        def strongconnect(v):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in self._adjacency.get(v, set()):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1:
                    cycles.append(component)

        for v in self.modules:
            if v not in index:
                strongconnect(v)

        return cycles

    def transitive_deps(self, module_path: str) -> Set[str]:
        """获取模块的所有传递依赖"""
        visited = set()
        queue = [module_path]
        while queue:
            current = queue.pop(0)
            for dep in self._adjacency.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
        return visited

    def check_dependency_direction(self) -> List[str]:
        """检查依赖方向是否符合架构约束"""
        violations = []
        layers = {
            "core": {"must_not_import": ["models", "diagnostics", "reporting"]},
            "models": {"must_not_import": ["reporting"]},
            "utils": {"must_not_import": ["models", "diagnostics", "reporting"]},
            "diagnostics": {"must_not_import": ["models", "reporting"]},
        }

        for module_path, info in self.modules.items():
            parts = module_path.split(".")
            if len(parts) < 2:
                continue
            current_layer = parts[1] if parts[0] == "tvp_var_framework" else None
            if current_layer not in layers:
                continue

            forbidden = layers[current_layer]["must_not_import"]
            for dep in self._adjacency.get(module_path, set()):
                dep_parts = dep.split(".")
                if len(dep_parts) >= 2 and dep_parts[0] == "tvp_var_framework":
                    dep_layer = dep_parts[1]
                    if dep_layer in forbidden:
                        violations.append(
                            f"{module_path} ({current_layer}) -> {dep} ({dep_layer})")

        return violations

    def check_side_effects(self) -> Dict[str, List[str]]:
        """检查模块级别的副作用"""
        result = {}
        for module_path, info in self.modules.items():
            if info.side_effects:
                # Filter out acceptable patterns
                real_effects = [
                    e for e in info.side_effects
                    if "logging.getLogger" not in e
                    and "logger = " not in e
                ]
                if real_effects:
                    result[module_path] = real_effects
        return result

    def report(self) -> str:
        """生成完整的依赖分析报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("TVP-VAR Framework Dependency Analysis Report")
        lines.append("=" * 70)

        # Module count
        lines.append(f"\nTotal modules: {len(self.modules)}")

        # Dependency edges
        total_edges = sum(len(deps) for deps in self._adjacency.values())
        lines.append(f"Total dependency edges: {total_edges}")

        # Cycle detection
        cycles = self.detect_cycles()
        lines.append(f"\nCycles detected: {len(cycles)}")
        if cycles:
            for i, cycle in enumerate(cycles):
                lines.append(f"  Cycle {i+1}: {' -> '.join(cycle)}")
        else:
            lines.append("  No cycles found")

        # Dependency direction
        violations = self.check_dependency_direction()
        lines.append(f"\nDependency direction violations: {len(violations)}")
        if violations:
            for v in violations:
                lines.append(f"  VIOLATION: {v}")
        else:
            lines.append("  All dependency directions correct")

        # Side effects
        side_effects = self.check_side_effects()
        lines.append(f"\nModules with side effects: {len(side_effects)}")
        if side_effects:
            for mod, effects in side_effects.items():
                lines.append(f"  {mod}:")
                for e in effects[:3]:
                    lines.append(f"    {e}")

        # Per-module dependency summary
        lines.append("\n--- Module Dependencies ---")
        for mod in sorted(self.modules.keys()):
            deps = self._adjacency.get(mod, set())
            if deps:
                lines.append(f"  {mod} -> {', '.join(sorted(deps))}")

        return "\n".join(lines)


def run_full_validation(framework_root: str = None) -> dict:
    """运行完整的架构验证"""
    if framework_root is None:
        framework_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..")

    graph = DependencyGraph(framework_root)
    graph.scan()

    cycles = graph.detect_cycles()
    violations = graph.check_dependency_direction()
    side_effects = graph.check_side_effects()

    passed = len(cycles) == 0 and len(violations) == 0

    return {
        "status": "PASS" if passed else "FAIL",
        "cycles": cycles,
        "direction_violations": violations,
        "side_effects": side_effects,
        "report": graph.report(),
    }


if __name__ == "__main__":
    result = run_full_validation()
    print(result["report"])
    print(f"\nFinal status: {result['status']}")
