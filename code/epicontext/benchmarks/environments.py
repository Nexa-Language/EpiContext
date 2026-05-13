"""
EpiContext Benchmark Environments

提供多种基准测试环境，用于评估EpiContext框架的性能。
"""

from __future__ import annotations

import json
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Abstract Environment
# ============================================================================

class BenchmarkEnvironment(ABC):
    """基准测试环境抽象基类。"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.step_count: int = 0
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def step(self, action: str) -> Tuple[str, bool]:
        """执行一步动作。

        Returns:
            (observation, success)
        """
        ...

    @abstractmethod
    def reset(self) -> str:
        """重置环境。

        Returns:
            初始观察
        """
        ...

    @abstractmethod
    def is_complete(self) -> bool:
        """检查任务是否完成。"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """获取环境统计信息。"""
        return {
            'step_count': self.step_count,
            'history_length': len(self.history),
        }


# ============================================================================
# WebArena-like Environment
# ============================================================================

@dataclass
class WebTask:
    """Web任务定义。"""
    task_id: str
    description: str
    required_steps: List[str]
    distractors: List[str] = field(default_factory=list)
    optimal_turns: int = 5


class WebArenaEnvironment(BenchmarkEnvironment):
    """WebArena风格的网页交互环境。

    模拟网页导航和操作任务，包含:
    - 多步操作序列
    - 干扰选项
    - 部分可观察性
    """

    WEB_TOOLS = [
        {'name': 'click', 'description': 'Click on a web element'},
        {'name': 'type', 'description': 'Type text into an input field'},
        {'name': 'scroll', 'description': 'Scroll the page'},
        {'name': 'navigate', 'description': 'Navigate to a URL'},
        {'name': 'search', 'description': 'Search within the page'},
        {'name': 'extract', 'description': 'Extract data from the page'},
        {'name': 'screenshot', 'description': 'Take a screenshot'},
        {'name': 'wait', 'description': 'Wait for page to load'},
        {'name': 'back', 'description': 'Go back to previous page'},
        {'name': 'forward', 'description': 'Go forward to next page'},
        {'name': 'refresh', 'description': 'Refresh the current page'},
        {'name': 'bookmark', 'description': 'Bookmark current page'},
        {'name': 'history', 'description': 'View browsing history'},
        {'name': 'download', 'description': 'Download a file'},
        {'name': 'upload', 'description': 'Upload a file'},
        {'name': 'fill_form', 'description': 'Fill out a web form'},
        {'name': 'select_option', 'description': 'Select from dropdown'},
        {'name': 'check_checkbox', 'description': 'Check a checkbox'},
        {'name': 'submit_form', 'description': 'Submit a form'},
        {'name': 'close_tab', 'description': 'Close current tab'},
    ]

    def __init__(self, task: WebTask, noise_level: float = 0.2, seed: int = 42):
        super().__init__(seed)
        self.task = task
        self.noise_level = noise_level
        self.completed_steps: List[str] = []
        self.current_page: str = 'home'
        self.page_content: Dict[str, str] = self._generate_page_content()

    def reset(self) -> str:
        """重置环境。"""
        self.step_count = 0
        self.completed_steps = []
        self.current_page = 'home'
        self.history = []
        return f"Current page: {self.current_page}. Task: {self.task.description}"

    def step(self, action: str) -> Tuple[str, bool]:
        """执行一步动作。"""
        self.step_count += 1
        success = False
        observation = ""

        # 检查动作是否匹配所需步骤
        for step in self.task.required_steps:
            if step.lower() in action.lower() and step not in self.completed_steps:
                self.completed_steps.append(step)
                success = True
                observation = f"Successfully executed: {step}"
                break

        if not success:
            # 添加噪声
            if self.rng.random() < self.noise_level:
                observation = f"Error: page element not found. Tried: {action[:50]}..."
            else:
                observation = f"Action completed but not required: {action[:50]}..."

        # 更新页面状态
        if 'navigate' in action.lower():
            self.current_page = f"page_{self.step_count}"

        self.history.append({
            'step': self.step_count,
            'action': action,
            'observation': observation,
            'success': success,
        })

        return observation, success

    def is_complete(self) -> bool:
        """检查是否所有必需步骤都已完成。"""
        return len(self.completed_steps) >= len(self.task.required_steps)

    def _generate_page_content(self) -> Dict[str, str]:
        """生成模拟页面内容。"""
        return {
            'home': '<html><body>Welcome to WebArena. Navigation menu available.</body></html>',
            'search': '<html><body>Search results for your query.</body></html>',
            'form': '<html><body>Please fill out the form below.</body></html>',
        }


# ============================================================================
# SWE-bench-like Environment
# ============================================================================

@dataclass
class SWETask:
    """软件工程任务定义。"""
    task_id: str
    description: str
    repo: str
    base_commit: str
    test_patch: str
    files_to_edit: List[str]
    difficulty: str = 'medium'  # 'easy', 'medium', 'hard'


class SWEBenchEnvironment(BenchmarkEnvironment):
    """SWE-bench风格的软件工程环境。

    模拟代码修复和功能实现任务。
    """

    SWE_TOOLS = [
        {'name': 'read_file', 'description': 'Read a file from the repository'},
        {'name': 'write_file', 'description': 'Write content to a file'},
        {'name': 'edit_file', 'description': 'Edit specific lines in a file'},
        {'name': 'search_code', 'description': 'Search for code patterns'},
        {'name': 'run_test', 'description': 'Run the test suite'},
        {'name': 'run_linter', 'description': 'Run code linter'},
        {'name': 'git_diff', 'description': 'Show git diff'},
        {'name': 'git_log', 'description': 'Show git log'},
        {'name': 'git_blame', 'description': 'Show git blame for a file'},
        {'name': 'find_definition', 'description': 'Find function/class definition'},
        {'name': 'find_references', 'description': 'Find all references to a symbol'},
        {'name': 'execute_command', 'description': 'Execute a shell command'},
        {'name': 'list_directory', 'description': 'List directory contents'},
        {'name': 'get_dependencies', 'description': 'List project dependencies'},
        {'name': 'check_syntax', 'description': 'Check Python syntax'},
        {'name': 'format_code', 'description': 'Format code with black/isort'},
        {'name': 'create_branch', 'description': 'Create a new git branch'},
        {'name': 'commit_changes', 'description': 'Commit changes to git'},
        {'name': 'view_issues', 'description': 'View related GitHub issues'},
        {'name': 'read_docs', 'description': 'Read project documentation'},
    ]

    def __init__(self, task: SWETask, seed: int = 42):
        super().__init__(seed)
        self.task = task
        self.files_edited: List[str] = []
        self.tests_passing: int = 0
        self.total_tests: int = 5
        self._file_contents: Dict[str, str] = self._generate_initial_files()

    def reset(self) -> str:
        """重置环境。"""
        self.step_count = 0
        self.files_edited = []
        self.tests_passing = 0
        self.history = []
        self._file_contents = self._generate_initial_files()
        return (
            f"Repository: {self.task.repo}\n"
            f"Task: {self.task.description}\n"
            f"Files to edit: {', '.join(self.task.files_to_edit)}"
        )

    def step(self, action: str) -> Tuple[str, bool]:
        """执行一步动作。"""
        self.step_count += 1
        success = False
        observation = ""

        # 检查是否编辑了目标文件
        for fname in self.task.files_to_edit:
            if fname in action and fname not in self.files_edited:
                self.files_edited.append(fname)
                success = True
                observation = f"File {fname} edited successfully."
                break

        if not success:
            if 'run_test' in action.lower():
                self.tests_passing = min(
                    self.total_tests,
                    self.tests_passing + 1,
                )
                observation = (
                    f"Tests: {self.tests_passing}/{self.total_tests} passing."
                )
                success = self.tests_passing >= self.total_tests
            elif 'search' in action.lower() or 'read' in action.lower():
                observation = f"Found relevant code for: {action[:50]}..."
                success = True
            else:
                observation = f"Action executed: {action[:50]}..."
                success = self.rng.random() > 0.3

        self.history.append({
            'step': self.step_count,
            'action': action,
            'observation': observation,
            'success': success,
        })

        return observation, success

    def is_complete(self) -> bool:
        """检查任务是否完成。"""
        return (
            len(self.files_edited) >= len(self.task.files_to_edit)
            and self.tests_passing >= self.total_tests
        )

    def _generate_initial_files(self) -> Dict[str, str]:
        """生成初始文件内容。"""
        contents = {}
        for fname in self.task.files_to_edit:
            contents[fname] = (
                f"# {fname}\n"
                f"# TODO: Fix issue described in {self.task.description}\n"
                f"def main():\n"
                f"    pass\n"
            )
        return contents


# ============================================================================
# ALFWorld-like Environment
# ============================================================================

@dataclass
class ALFTask:
    """ALFWorld任务定义。"""
    task_id: str
    description: str
    target_object: str
    target_location: str
    required_actions: List[str]
    room_layout: Dict[str, List[str]] = field(default_factory=dict)


class ALFWorldEnvironment(BenchmarkEnvironment):
    """ALFWorld风格的家庭环境文本游戏。

    模拟家庭环境中的导航和操作任务。
    """

    ALF_TOOLS = [
        {'name': 'go_to', 'description': 'Go to a specific location'},
        {'name': 'take', 'description': 'Take an object'},
        {'name': 'put', 'description': 'Put an object somewhere'},
        {'name': 'open', 'description': 'Open a container'},
        {'name': 'close', 'description': 'Close a container'},
        {'name': 'look', 'description': 'Look around the current room'},
        {'name': 'inventory', 'description': 'Check your inventory'},
        {'name': 'examine', 'description': 'Examine an object closely'},
        {'name': 'use', 'description': 'Use an object'},
        {'name': 'clean', 'description': 'Clean an object'},
        {'name': 'heat', 'description': 'Heat an object'},
        {'name': 'cool', 'description': 'Cool an object'},
        {'name': 'slice', 'description': 'Slice an object'},
        {'name': 'search', 'description': 'Search the current location'},
        {'name': 'read', 'description': 'Read a note or book'},
        {'name': 'turn_on', 'description': 'Turn on a device'},
        {'name': 'turn_off', 'description': 'Turn off a device'},
        {'name': 'pick_up', 'description': 'Pick up an object from the floor'},
        {'name': 'drop', 'description': 'Drop an object from inventory'},
        {'name': 'push', 'description': 'Push an object'},
    ]

    ROOMS = ['kitchen', 'living_room', 'bedroom', 'bathroom', 'hallway']
    OBJECTS = ['apple', 'book', 'cup', 'key', 'lamp', 'plate', 'soap', 'towel']

    def __init__(self, task: ALFTask, seed: int = 42):
        super().__init__(seed)
        self.task = task
        self.current_room: str = 'kitchen'
        self.inventory: List[str] = []
        self.room_objects: Dict[str, List[str]] = self._generate_room_objects()
        self.completed_actions: List[str] = []

    def reset(self) -> str:
        """重置环境。"""
        self.step_count = 0
        self.current_room = 'kitchen'
        self.inventory = []
        self.completed_actions = []
        self.history = []
        self.room_objects = self._generate_room_objects()
        return (
            f"You are in the {self.current_room}. "
            f"Task: {self.task.description}\n"
            f"You see: {', '.join(self.room_objects.get(self.current_room, []))}"
        )

    def step(self, action: str) -> Tuple[str, bool]:
        """执行一步动作。"""
        self.step_count += 1
        success = False
        observation = ""

        action_lower = action.lower()

        # 导航
        if 'go_to' in action_lower:
            for room in self.ROOMS:
                if room in action_lower:
                    self.current_room = room
                    success = True
                    observation = (
                        f"You are now in the {room}. "
                        f"You see: {', '.join(self.room_objects.get(room, []))}"
                    )
                    break
            if not success:
                observation = f"Cannot go there from {self.current_room}."

        # 拾取物品
        elif 'take' in action_lower or 'pick_up' in action_lower:
            for obj in self.room_objects.get(self.current_room, []):
                if obj in action_lower:
                    self.room_objects[self.current_room].remove(obj)
                    self.inventory.append(obj)
                    success = True
                    observation = f"You picked up the {obj}."
                    break
            if not success:
                observation = "That object is not here."

        # 检查任务完成
        elif any(req.lower() in action_lower for req in self.task.required_actions):
            if action_lower not in self.completed_actions:
                self.completed_actions.append(action_lower)
                success = True
                observation = f"Action completed: {action[:50]}..."
            else:
                observation = "Already done that."

        else:
            observation = f"You try to {action[:50]}... Nothing happens."
            success = False

        self.history.append({
            'step': self.step_count,
            'action': action,
            'observation': observation,
            'success': success,
            'room': self.current_room,
            'inventory': list(self.inventory),
        })

        return observation, success

    def is_complete(self) -> bool:
        """检查任务是否完成。"""
        return len(self.completed_actions) >= len(self.task.required_actions)

    def _generate_room_objects(self) -> Dict[str, List[str]]:
        """生成房间物品布局。"""
        objects = {}
        available = list(self.OBJECTS)
        self.rng.shuffle(available)

        for room in self.ROOMS:
            n_objects = self.rng.randint(1, 3)
            objects[room] = []
            for _ in range(n_objects):
                if available:
                    objects[room].append(available.pop())

        # 确保目标物品在目标位置
        if self.task.target_object not in sum(objects.values(), []):
            objects[self.task.target_location].append(self.task.target_object)

        return objects


# ============================================================================
# AgentBench-like Environment
# ============================================================================

@dataclass
class AgentBenchTask:
    """AgentBench任务定义。"""
    task_id: str
    description: str
    category: str  # 'code', 'web', 'os', 'db', 'reasoning'
    subtasks: List[Dict[str, Any]]
    tools_required: List[str]


class AgentBenchEnvironment(BenchmarkEnvironment):
    """AgentBench风格的多领域Agent环境。"""

    AGENTBENCH_TOOLS = [
        {'name': 'bash', 'description': 'Execute a bash command'},
        {'name': 'python', 'description': 'Execute Python code'},
        {'name': 'sql', 'description': 'Execute SQL query'},
        {'name': 'web_search', 'description': 'Search the web'},
        {'name': 'web_browse', 'description': 'Browse a web page'},
        {'name': 'file_read', 'description': 'Read a file'},
        {'name': 'file_write', 'description': 'Write to a file'},
        {'name': 'file_list', 'description': 'List files in directory'},
        {'name': 'api_call', 'description': 'Make an API call'},
        {'name': 'calculator', 'description': 'Perform calculations'},
        {'name': 'translate', 'description': 'Translate text'},
        {'name': 'summarize', 'description': 'Summarize text'},
        {'name': 'classify', 'description': 'Classify text'},
        {'name': 'extract_entities', 'description': 'Extract named entities'},
        {'name': 'sentiment', 'description': 'Analyze sentiment'},
        {'name': 'code_review', 'description': 'Review code for issues'},
        {'name': 'generate_code', 'description': 'Generate code from description'},
        {'name': 'debug', 'description': 'Debug code'},
        {'name': 'test', 'description': 'Run tests'},
        {'name': 'deploy', 'description': 'Deploy application'},
    ]

    def __init__(self, task: AgentBenchTask, seed: int = 42):
        super().__init__(seed)
        self.task = task
        self.completed_subtasks: List[int] = []
        self.current_subtask: int = 0

    def reset(self) -> str:
        """重置环境。"""
        self.step_count = 0
        self.completed_subtasks = []
        self.current_subtask = 0
        self.history = []
        subtask = self.task.subtasks[0] if self.task.subtasks else {}
        return (
            f"Category: {self.task.category}\n"
            f"Task: {self.task.description}\n"
            f"Current subtask: {subtask.get('description', 'N/A')}"
        )

    def step(self, action: str) -> Tuple[str, bool]:
        """执行一步动作。"""
        self.step_count += 1
        success = False
        observation = ""

        if self.current_subtask < len(self.task.subtasks):
            subtask = self.task.subtasks[self.current_subtask]
            expected = subtask.get('expected_action', '')

            if expected.lower() in action.lower():
                self.completed_subtasks.append(self.current_subtask)
                self.current_subtask += 1
                success = True
                observation = f"Subtask {self.current_subtask} completed."

                if self.current_subtask < len(self.task.subtasks):
                    next_subtask = self.task.subtasks[self.current_subtask]
                    observation += f" Next: {next_subtask.get('description', '')}"
            else:
                observation = f"Action not matching expected: {expected[:50]}..."
        else:
            observation = "All subtasks completed."
            success = True

        self.history.append({
            'step': self.step_count,
            'action': action,
            'observation': observation,
            'success': success,
            'subtask': self.current_subtask,
        })

        return observation, success

    def is_complete(self) -> bool:
        """检查所有子任务是否完成。"""
        return self.current_subtask >= len(self.task.subtasks)


# ============================================================================
# Environment Factory
# ============================================================================

def create_webarena_tasks(num_tasks: int = 10, seed: int = 42) -> List[WebTask]:
    """创建WebArena任务集。"""
    rng = np.random.RandomState(seed)
    tasks = []

    templates = [
        {
            'prefix': 'Navigate to the profile page and update',
            'steps': ['navigate', 'click', 'type', 'submit_form'],
            'distractors': ['scroll', 'bookmark', 'screenshot'],
        },
        {
            'prefix': 'Search for a product and add to',
            'steps': ['search', 'click', 'select_option', 'click'],
            'distractors': ['back', 'refresh', 'history'],
        },
        {
            'prefix': 'Fill out the contact form with',
            'steps': ['navigate', 'type', 'type', 'type', 'submit_form'],
            'distractors': ['download', 'upload', 'screenshot'],
        },
        {
            'prefix': 'Find the latest news article about',
            'steps': ['navigate', 'search', 'click', 'extract'],
            'distractors': ['bookmark', 'wait', 'scroll'],
        },
        {
            'prefix': 'Download the report and extract',
            'steps': ['navigate', 'click', 'download', 'extract'],
            'distractors': ['back', 'forward', 'refresh'],
        },
    ]

    for i in range(num_tasks):
        template = templates[i % len(templates)]
        tasks.append(WebTask(
            task_id=f'webarena_{i:03d}',
            description=f"{template['prefix']} task {i}.",
            required_steps=template['steps'],
            distractors=template['distractors'],
            optimal_turns=len(template['steps']) + rng.randint(0, 3),
        ))

    return tasks


def create_swebench_tasks(num_tasks: int = 10, seed: int = 42) -> List[SWETask]:
    """创建SWE-bench任务集。"""
    rng = np.random.RandomState(seed)
    tasks = []

    repos = ['django/django', 'pytest-dev/pytest', 'scikit-learn/scikit-learn',
             'pallets/flask', 'psf/requests']
    difficulties = ['easy', 'medium', 'hard']

    for i in range(num_tasks):
        repo = repos[i % len(repos)]
        difficulty = difficulties[i % len(difficulties)]
        tasks.append(SWETask(
            task_id=f'swebench_{i:03d}',
            description=f"Fix bug #{1000+i} in {repo}: handle edge case in input validation.",
            repo=repo,
            base_commit=f"abc{i:04d}",
            test_patch=f"test_fix_{i}.py",
            files_to_edit=[f"src/module_{j}.py" for j in range(1, rng.randint(2, 4))],
            difficulty=difficulty,
        ))

    return tasks


def create_alfworld_tasks(num_tasks: int = 10, seed: int = 42) -> List[ALFTask]:
    """创建ALFWorld任务集。"""
    rng = np.random.RandomState(seed)
    tasks = []

    for i in range(num_tasks):
        target_obj = ALFWorldEnvironment.OBJECTS[i % len(ALFWorldEnvironment.OBJECTS)]
        target_room = ALFWorldEnvironment.ROOMS[(i + 1) % len(ALFWorldEnvironment.ROOMS)]

        tasks.append(ALFTask(
            task_id=f'alfworld_{i:03d}',
            description=f"Find the {target_obj} and put it on the table.",
            target_object=target_obj,
            target_location=target_room,
            required_actions=[
                f'go_to {target_room}',
                f'take {target_obj}',
                'go_to kitchen',
                f'put {target_obj}',
            ],
            room_layout={},
        ))

    return tasks


def create_agentbench_tasks(num_tasks: int = 10, seed: int = 42) -> List[AgentBenchTask]:
    """创建AgentBench任务集。"""
    rng = np.random.RandomState(seed)
    tasks = []

    categories = ['code', 'web', 'os', 'db', 'reasoning']

    for i in range(num_tasks):
        cat = categories[i % len(categories)]
        num_subtasks = rng.randint(2, 5)

        subtasks = []
        for j in range(num_subtasks):
            subtasks.append({
                'id': j,
                'description': f"Subtask {j+1} for {cat} task {i}",
                'expected_action': f"action_{cat}_{i}_{j}",
            })

        tasks.append(AgentBenchTask(
            task_id=f'agentbench_{i:03d}',
            description=f"Complete {cat} task #{i}: process and analyze data.",
            category=cat,
            subtasks=subtasks,
            tools_required=['bash', 'python', 'file_read', 'file_write'],
        ))

    return tasks


def create_environment_factory(
    task_type: str,
    tasks: List[Any],
    seed: int = 42,
) -> Callable[[str], BenchmarkEnvironment]:
    """创建环境工厂函数。

    Args:
        task_type: 任务类型 ('webarena', 'swebench', 'alfworld', 'agentbench')
        tasks: 任务列表
        seed: 随机种子

    Returns:
        环境工厂函数
    """
    task_map = {t.task_id: t for t in tasks}

    def factory(task_id: str) -> BenchmarkEnvironment:
        task = task_map[task_id]
        if task_type == 'webarena':
            return WebArenaEnvironment(task, seed=seed)
        elif task_type == 'swebench':
            return SWEBenchEnvironment(task, seed=seed)
        elif task_type == 'alfworld':
            return ALFWorldEnvironment(task, seed=seed)
        elif task_type == 'agentbench':
            return AgentBenchEnvironment(task, seed=seed)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    return factory


def get_tools_for_type(task_type: str) -> List[Dict[str, Any]]:
    """获取指定任务类型的工具列表。"""
    tool_map = {
        'webarena': WebArenaEnvironment.WEB_TOOLS,
        'swebench': SWEBenchEnvironment.SWE_TOOLS,
        'alfworld': ALFWorldEnvironment.ALF_TOOLS,
        'agentbench': AgentBenchEnvironment.AGENTBENCH_TOOLS,
    }
    return tool_map.get(task_type, [])