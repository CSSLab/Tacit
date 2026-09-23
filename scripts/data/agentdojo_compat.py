"""agentdojo 0.1.35 with pydantic 2.13: rebuild TaskResults, whose deferred FunctionCall
annotation otherwise fails when cached episode results are loaded. Imported through the
benchmark CLI's module hook (-ml agentdojo_compat)."""
from agentdojo.functions_runtime import FunctionCall  # noqa: F401
from agentdojo.benchmark import TaskResults

TaskResults.model_rebuild()
