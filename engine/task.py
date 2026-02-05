import random
import string
from collections import defaultdict, deque

RELATIONS = [
    "taller than",
    "older than",
    "faster than",
    "heavier than",
    "larger than"
]

REL_WORDS = [
    ("left of", "right of"),
    ("above", "below"),
    ("faster than", "slower than")
]

def comparison_params_from_difficulty(log_difficulty):
    if log_difficulty < 8:
        return {
            "min_depth": 2,
            "min_query_dist":  1,
            "noise_edges": 0,
            "no_of_objects": 4
        }
    elif log_difficulty < 15:
        return {
            "min_depth": 3,
            "min_query_dist":  2,
            "noise_edges": 1,
            "no_of_objects": random.randint(5, 6)
        }
    elif log_difficulty < 25:
        return {
            "min_depth": 4,
            "min_query_dist":  3,
            "noise_edges": 2,
            "no_of_objects": random.randint(7, 8)
        }
    else:
        return {
            "min_depth": 5,
            "min_query_dist":  4,
            "noise_edges": 3,
            "no_of_objects": random.randint(9, 10)
        }

def boolean_params_from_difficulty(log_difficulty):
    if log_difficulty < 5:
        var_count = 2
        ops_count = 1
        allow_not = False
    elif log_difficulty < 10:
        var_count = random.choice([2, 3])
        ops_count = 2
        allow_not = False
    elif log_difficulty < 15:
        var_count = 3
        ops_count = 2
        allow_not = True
    elif log_difficulty < 20:
        var_count = random.choice([3, 4])
        ops_count = random.choice([2, 3])
        allow_not = True
    elif log_difficulty < 25:
        var_count = random.choice([4, 5])
        ops_count = random.choice([3, 4])
        allow_not = True
    else:
        var_count = random.choice([4, 6])
        ops_count = random.choice([3, 5])
        allow_not = True
    return var_count, ops_count, allow_not

def implication_params_from_difficulty(log_difficulty):
    if log_difficulty < 5:
        vars_count = 3
        edges = 2
    elif log_difficulty < 10:
        vars_count = 4
        edges = 3
    elif log_difficulty < 15:
        vars_count = random.choice([4, 5])
        edges = random.choice([3, 4])
    elif log_difficulty < 20:
        vars_count = random.choice([5, 6])
        edges = random.choice([4, 5])
    elif log_difficulty < 25:
        vars_count = random.choice([6, 7])
        edges = random.choice([5, 6])
    else:
        vars_count = random.choice([7, 8, 9])
        edges = random.choice([6, 7, 8])
    return vars_count, edges

def constraint_params_from_difficulty(log_difficulty):
    if log_difficulty < 5:
        n_objects = 3
        n_constraints = 2
    elif log_difficulty < 10:
        n_objects = 4
        n_constraints = 3
    elif log_difficulty < 15:
        n_objects = random.choice([4, 5])
        n_constraints = random.choice([3, 4])
    elif log_difficulty < 20:
        n_objects = random.choice([5, 6])
        n_constraints = random.choice([4, 5])
    elif log_difficulty < 25:
        n_objects = random.choice([6, 7])
        n_constraints = random.choice([5, 6])
    else:
        n_objects = random.choice([7, 8, 9])
        n_constraints = random.choice([6, 7, 8])
    return n_objects, n_constraints

def ap_params_from_difficulty(pat_difficulty):
    if pat_difficulty < 5:
        length = 4
        step_max = 5
    elif pat_difficulty < 10:
        length = 5
        step_max = 7
    elif pat_difficulty < 15:
        length = random.choice([5, 6])
        step_max = 10
    elif pat_difficulty < 20:
        length = random.choice([6, 7])
        step_max = 15
    elif pat_difficulty < 25:
        length = random.choice([7, 8])
        step_max = random.choice([15, 20])
    else:
        length = random.choice([9, 10, 11, 12, 13])
        step_max = random.choice([25, 30, 35])
    return length, step_max

def mp_params_from_difficulty(pat_difficulty):
    if pat_difficulty < 5:
        length = random.choice([7, 8, 9])
        multipliers = [2]
        start = random.randint(1, 10)
    elif pat_difficulty < 10:
        length = random.choice([6, 7, 8])
        multipliers = [2]
        start = random.randint(3, 14)
    elif pat_difficulty < 15:
        length = random.choice([5, 6, 7])
        multipliers = [2, 3]
        start = random.randint(5, 15)
    elif pat_difficulty < 20:
        length = random.choice([4, 5, 6])
        multipliers = [2, 3]
        start = random.randint(7, 17)
    elif pat_difficulty < 25:
        length = random.choice([3, 4, 5])
        multipliers = [3, 4]
        start = random.randint(10, 20)
    else:
        length = random.choice([3, 4])
        multipliers = [3, 4, 5]
        start = random.randint(10, 25)
    return length, multipliers, start

def cycle_params_from_difficulty(pat_difficulty):
    if pat_difficulty < 5:
        cycle_len = 2
        total_len = random.randint(4, 5)
    elif pat_difficulty < 10:
        cycle_len = 3
        total_len = random.randint(6, 7)
    elif pat_difficulty < 15:
        cycle_len = random.choice([3, 4])
        total_len = random.randint(7, 9)
    elif pat_difficulty < 20:
        cycle_len = random.choice([4, 5])
        total_len = random.randint(8, 11)
    elif pat_difficulty < 25:
        cycle_len = random.choice([5, 6])
        total_len = random.randint(10, 13)
    else:
        cycle_len = random.choice([6, 7, 8])
        total_len = random.randint(12, 20)
    return cycle_len, total_len

def mixed_params_from_difficulty(pat_difficulty):
    if pat_difficulty < 5:
        op_len = 2
        seq_len = random.randint(5, 6)
        k_max = 3
    elif pat_difficulty < 10:
        op_len = 2
        seq_len = random.randint(6, 7)
        k_max = 5
    elif pat_difficulty < 15:
        op_len = random.choice([2, 3])
        seq_len = random.randint(7, 8)
        k_max = 6
    elif pat_difficulty < 20:
        op_len = random.choice([2, 3])
        seq_len = random.randint(8, 9)
        k_max = 8
    elif pat_difficulty < 25:
        op_len = random.choice([2, 4])
        seq_len = random.randint(8, 10)
        k_max = 10
    else:
        op_len = random.choice([3, 4])
        seq_len = random.randint(10, 13)
        k_max = 12
    return op_len, seq_len, k_max

def division_params_from_difficulty(num_diff):
    if num_diff < 8:
        return {
            "base_divisors": [4, 5, 6, 8],
            "answer_range": (2, 10),
            "simplify_steps": 1
        }
    elif num_diff < 16:
        return {
            "base_divisors": [6, 8, 9, 12, 15],
            "answer_range": (3, 12),
            "simplify_steps": 2
        }
    elif num_diff < 25:
        return {
            "base_divisors": [12, 15, 18, 20, 24],
            "answer_range": (4, 15),
            "simplify_steps": 2
        }
    else:
        return {
            "base_divisors": [18, 20, 24, 30, 36, 42],
            "answer_range": (5, 18),
            "simplify_steps": 3
        }
    
def ratio_params_from_difficulty(num_diff):
    if num_diff < 5:
        return {
            "k": random.randint(2, 3),
            "a_range": (2, 8),
            "mode": "multiply"
        }
    elif num_diff < 10:
        return {
            "k": random.randint(3, 5),
            "a_range": (4, 10),
            "mode": "multiply"
        }
    elif num_diff < 15:
        return {
            "k": random.randint(3, 7),
            "a_range": (6, 15),
            "mode": random.choice(["multiply", "divide"])
        }
    elif num_diff < 20:
        return {
            "k": random.randint(4, 7),
            "a_range": (8, 20),
            "mode": random.choice(["multiply", "divide"])
        }
    else:
        return {
            "k": random.randint(4, 8),
            "a_range": (10, 25),
            "mode": random.choice(["multiply", "divide"])
        }

def square_params_from_difficulty(num_diff):
    if num_diff < 5:
        return {
            "base_range": (2,9),
            "offsets": [0],
            "mode": "exact"
        }
    elif num_diff < 10:
        return {
            "base_range": (5, 15),
            "offsets": [0, 1],
            "mode": "near"
        }
    elif num_diff < 15:
        return {
            "base_range": (10, 20),
            "offsets": [1, 2],
            "mode": "near"
        }
    elif num_diff < 20:
        return {
            "base_range": (15, 25),
            "offsets": [1, 2, 3],
            "mode": "near"
        }
    else:
        return {
            "base_range": (25, 40),
            "offsets": [1, 2, 3, 5],
            "mode": "decompose"
        }
    
def near_boundary_param_from_difficulty(num_diff):
    if num_diff < 5:
        boundary = [10, 50]
        offset = 5
        ops = ["+"]
    elif num_diff < 10:
        boundary = [50, 100]
        offset = 10
        ops = ["+"]
    elif num_diff < 15:
        boundary = [100, 200]
        offset = 15
        ops = ["+", "-"]
    elif num_diff < 20:
        boundary = [200, 500]
        offset = 20
        ops = ["+", "-"]
    elif num_diff < 25:
        boundary = [500, 1000]
        offset = 30
        ops = ["+", "-"]
    else:
        boundary = [1000, 2000]
        offset = 50
        ops = ["+", "-"]
    return boundary, offset, ops
  
def graph_depth(graph):
    memo = {}
    def dfs(node, visiting=None):
        if visiting is None:
            visiting = set()
        if node in memo:
            return memo[node]
        if node in visiting:
            return 1
        visiting.add(node)
        if not graph[node]:
            result = 1
        else:
            result = 1 + max(dfs(n, visiting.copy()) for n in graph[node])
        memo[node] = result
        return result
    
    if not graph:
        return 0
    return max(dfs(n) for n in list(graph.keys()))

def shortest_path_length(graph, start, end):
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None

def is_arithmetic(seq):
    if len(seq) < 3:
        return False
    d = seq[1] - seq[0]
    return all(seq[i+1] - seq[i] == d for i in range(len(seq)-1))

def is_multiplicative(seq):
    if len(seq) < 3 or 0 in seq:
        return False
    r = seq[1] / seq[0]
    return all(seq[i+1] / seq[i] == r for i in range(len(seq)-1))

def eval_boolean(expr, values):
    if isinstance(expr, str):
        return values[expr]
    if expr[0] == "NOT":
        return not eval_boolean(expr[1], values)
    op, left, right = expr
    if op == "AND":
        return eval_boolean(left, values) and eval_boolean(right, values)
    elif op == "OR":
        return eval_boolean(left, values) or eval_boolean(right, values)

def build_boolean_expression(vars_list, ops_count, allow_not):
    expr = random.choice(vars_list)
    not_count = 0
    op_count = 0
    for i in range(ops_count):
        op = random.choice(["AND", "OR"])
        right = random.choice(vars_list)
        if allow_not and random.choice([True, False]):
            right = ("NOT", right)
            not_count += 1
        expr = (op, expr, right)
        op_count += 1
    if allow_not and random.choice([True, False]):
        expr = ("NOT", expr)
        not_count += 1
    return expr, op_count, not_count

def render_boolean(expr):
    if isinstance(expr, str):
        return expr

    if expr[0] == "NOT":
        inner = expr[1]
        if isinstance(inner, str):
            return f"(NOT {inner})"
        else:
            return f"(NOT {render_boolean(inner)})"
    op, left, right = expr
    return f"({render_boolean(left)} {op} {render_boolean(right)})"

def propagate_truth(graph, true_sets):
    queue = deque(true_sets)
    inferred = set(true_sets)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in inferred:
                inferred.add(neighbor)
                queue.append(neighbor)
    return inferred

def is_relevant(base, graph):
    return base in graph or any(base in nbrs for nbrs in graph.values())

def build_graph(comparisons):
    graph = defaultdict(set)
    
    for a, op, b in comparisons:
        if op == ">":
            graph[a].add(b)
        elif op == "<":
            graph[b].add(a)
    return graph

def has_path(graph, start, end):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

def shortest_inference_depth(graph, base_true, target):
    queue = deque([(v, 0) for v in base_true])
    visited = set(base_true)
    while queue:
        node, depth = queue.popleft()
        if node == target:
            return depth
        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return None

def generate_comparison(log_difficulty):
    while True:
        spec = comparison_params_from_difficulty(log_difficulty)
        no_of_objects = spec["no_of_objects"]
        objects = list(string.ascii_uppercase[:no_of_objects])
        min_depth = spec["min_depth"]
        min_query_dist = spec["min_query_dist"]
        noise_edges = spec["noise_edges"]
        relation_word = random.choice(RELATIONS)
        comparisons = set()
        graph = defaultdict(set)
        target_edges = max(2, no_of_objects - 1) 
        attemps = 0
        while len(comparisons) < target_edges and attemps < 200:
            a, b = random.sample(objects, 2)
            attemps += 1
            if a == b:
                continue
            if has_path(graph, b, a):  
                continue
            if len(graph[a]) >= 2:    
                continue
            comparisons.add((a, ">", b))
            graph[a].add(b)
        graph = build_graph(comparisons)
        depth = graph_depth(graph)
        if depth < min_depth:
            continue
        target_answer = random.choice(["yes", "no"])
        query = None 
        query_dist = None
        for _ in range(1000):
            x, y = random.sample(objects, 2)
            dist = shortest_path_length(graph, x, y)
            if target_answer == "yes":
                if dist is None or dist < min_query_dist:
                    continue
            else:
                if dist is not None:
                    continue
            if (x, ">", y) in comparisons or  (y, ">", x) in comparisons:
                continue
            query = (x, y)
            query_dist = dist
            break
        if query is None:
            continue
        x, y = query
        noise_added = 0
        attemps = 0
        while noise_added < noise_edges and attemps < 200:
            a, b = random.sample(objects, 2)
            attemps += 1
            if has_path(graph, a, b) or has_path(graph, b, a):
                continue
            comparisons.add((a, ">", b))
            graph[a].add(b)
            noise_added += 1
        answer = "yes" if has_path(graph, x, y) else "no"
        lines = []
        for a, _, b in comparisons:
            lines.append(f"{a} is {relation_word} {b}.\n")
        lines.append(f"Is {x} {relation_word} {y}?")
        question_text = " ".join(lines)
        return {
            "task_id": f"LOG-COMP-{log_difficulty}-{no_of_objects}",
            "skill": "logical",
            "difficulty": log_difficulty,
            "question": question_text,
            "answer": answer,
            "answer_type": "yesno",
            "comparison_objects": no_of_objects,
            "comparison_edges": len(comparisons),
            "comparison_depth": depth,
            "query_distance": query_dist,
            "noise_edges": noise_edges
        }

def generate_boolean_evaluation(log_difficulty):
    vars_count, ops_count, allow_not = boolean_params_from_difficulty(log_difficulty)
    variables = list(string.ascii_uppercase[:vars_count])
    values = {v: random.choice([True, False]) for v in variables}
    expr, op_count, not_count = build_boolean_expression(variables, ops_count, allow_not)
    result = eval_boolean(expr, values)
    lines = []
    for v in variables:
        lines.append(f"{v} is {values[v]}.")
    lines.append(f"What is: {render_boolean(expr)}?")
    question_text = "\n".join(lines)
    return {
        "task_id": f"LOG-BOOL-{log_difficulty}-{vars_count}-{ops_count}",
        "skill": "logical",
        "difficulty": log_difficulty,
        "question": question_text,
        "answer": "True" if result else "False",
        "answer_type": "boolean",
        "boolean_var_count": vars_count,
        "boolean_op_count": op_count,
        "boolean_not_count": not_count
    }
 
def generate_implication_chain(log_difficulty):
    vars_count, edges = implication_params_from_difficulty(log_difficulty)
    variables = list(string.ascii_uppercase[:vars_count])
    while True: 
        graph = defaultdict(set)
        all_edges = set()
        while len(all_edges) < edges:
            a, b = random.sample(variables, 2)
            if a != b:
                all_edges.add((a, b))
        for a, b in all_edges:
            graph[a].add(b)
        base_true = set(random.sample(variables, random.randint(1, 2)))
        if not any(is_relevant(b, graph) for b in base_true):
            continue
        inferred = propagate_truth(graph, base_true)
        candidates = [
            v for v in variables
            if v not in base_true and is_relevant(v, graph)
        ]
        if not candidates:
            continue
        query = random.choice(candidates)
        chain_depth = shortest_inference_depth(graph, base_true, query)
        answer = "yes" if query in inferred else "no"
        lines = []
        for v in base_true:
            lines.append(f"{v} is True.")
        for a, b in all_edges:
            lines.append(f"If {a} is True, then {b} is True.")
        lines.append(f"Is {query} True?")
        question_text = "\n".join(lines)
        break
    return {
        "task_id": f"LOG-IMP-{log_difficulty}-{vars_count}-{edges}",
        "skill": "logical",
        "difficulty": log_difficulty,
        "question": question_text,
        "answer": answer,
        "answer_type": "yesno",
        "imp_var_count": vars_count,
        "imp_edge_count": edges,
        "imp_base_true_count": len(base_true),
        "imp_chain_depth": chain_depth
    }
   
def generate_constraint_task(log_difficulty):
    n_objects, n_constraints = constraint_params_from_difficulty(log_difficulty)
    objects = list(string.ascii_uppercase[:n_objects])
    relation, inverse = random.choice(REL_WORDS)
    graph = defaultdict(set)
    constraints = set()
    attempts = 0
    while len(constraints) < n_constraints and attempts < 500:
        a, b = random.sample(objects, 2)
        attempts += 1
        if has_path(graph, a, b):
            continue
        if (a, b) in constraints:
            continue
        if (b, a) in constraints:
            continue
        constraints.add((a, b))
        graph[a].add(b)
    depth = graph_depth(graph)
    target_answer = random.choice(["yes", "no"])
    query = None
    for _ in range(1000):
        x, y = random.sample(objects, 2)
        if x == y:
            continue
        reachable = has_path(graph, x, y)
        if target_answer == "yes" and reachable:
            query = (x, y)
            break
        if target_answer == "no" and not reachable:
            query = (x, y)
            break
    if query is None:
        return generate_constraint_task(log_difficulty)
    if log_difficulty >= 10:
        if (x, y) in constraints or (y, x) in constraints:
            return generate_constraint_task(log_difficulty)
    if log_difficulty >= 15:
        if shortest_path_length(graph, x, y) is not None and shortest_path_length(graph, x, y) < 2:
            return generate_constraint_task(log_difficulty)
    x, y = query
    answer = "yes" if has_path(graph, x, y) else "no"
    lines = []
    for a, b in constraints:
        lines.append(f"{a} is {relation} {b}.")
    lines.append(f"Is {x} {relation} {y}?")
    return {
        "task_id": f"LOG-CONSTRAINT-{log_difficulty}-{n_objects}-{n_constraints}",
        "skill": "logical",
        "difficulty": log_difficulty,
        "question": "\n".join(lines),
        "answer": answer,
        "answer_type": "yesno",
        "constraint_objects": n_objects,
        "constraint_count": n_constraints,
        "constraint_depth": depth
    }

def generate_multiplicative_progression(pat_difficulty):
    length, multipliers, start = mp_params_from_difficulty(pat_difficulty)
    multiplier = random.choice(multipliers)
    seq = [start * (multiplier ** i) for i in range(length)]
    answer = seq[-1] * multiplier
    if answer > 100000:
        return generate_multiplicative_progression(pat_difficulty)
    seq.append("?")
    seq_text = ", ".join(str(x) for x in seq)

    return {
        "task_id": f"PAT-MP-{pat_difficulty}-{length}-{multiplier}",
        "skill": "pattern",
        "difficulty": pat_difficulty,
        "question": seq_text,
        "answer": answer,
        "answer_type": "integer",
        "mp_multiplier": multiplier,
        "mp_length": length
    }

def generate_arithmetic_progression(pat_difficulty):
    while True:
        length, step_max = ap_params_from_difficulty(pat_difficulty)
        step = random.choice([i for i in range(-step_max, step_max + 1) if i != 0])
        if pat_difficulty >= 5 and abs(step) <= 2:
            continue
        if pat_difficulty < 10:
            start = random.randint(0, 50)
        else:
            start = random.randint(-50, 50)
        if pat_difficulty >= 15 and start % step == 0:
            continue
        seq = [start + i * step for i in range(length)]
        answer = seq[-1] + step
        seq.append("?")
        seq_text = ", ".join(str(x) for x in seq)
        return {
            "task_id": f"PAT-AP-{pat_difficulty}-{length}-{step_max}",
            "skill": "pattern",
            "difficulty": pat_difficulty,
            "question": seq_text,
            "answer": answer,
            "answer_type": "integer",
            "ap_length": length,
            "ap_step": step,
            "ap_abs_step": abs(step)
        }
 
def generate_cycle_pattern(pat_difficulty):
    cycle_len, total_len = cycle_params_from_difficulty(pat_difficulty)
    while True:
        cycle = [random.randint(1, 20) for _ in range(cycle_len)]
        if is_arithmetic(cycle) or is_multiplicative(cycle):
            continue
        break
    if pat_difficulty >= 18:
        offset = random.randint(1, cycle_len-1)
    else: 
        offset = 0
    seq = []
    for i in range(total_len):
        seq.append(cycle[(offset + i) % cycle_len])
        
    answer = cycle[(offset + total_len) % cycle_len]
    seq.append("?")
    seq_text = ", ".join(str(x) for x in seq)
    return {
        "task_id": f"PAT-CP-{pat_difficulty}-{cycle_len}-{total_len}",
        "skill": "pattern",
        "difficulty": pat_difficulty,
        "question": seq_text,
        "answer": answer,
        "answer_type": "integer",
        "cycle_len": cycle_len,
        "cycle_offset": offset,
        "cycle_total_len": total_len
    }

def generate_mixed_pattern(pat_difficulty):
    op_len, seq_len, k_max = mixed_params_from_difficulty(pat_difficulty)
    while True:
        ops = []
        for i in range(op_len):
            op_type = random.choice(["+", "*"])
            k = random.randint(2, k_max)
            ops.append((op_type, k))
        current = random.randint(1, 20)
        seq = [current]
        for i in range(1, seq_len):
            op, k = ops[(i - 1) % op_len]
            if op == "+":
                current += k
            elif op == "*":
                if op == "*" and k >= 5 and pat_difficulty < 25:
                    continue
                current *= k
            if current <= 0 or current > 10000:
                break
            seq.append(current)
        if len(seq) < seq_len:
            continue
        if is_arithmetic(seq) or is_multiplicative(seq):
            continue
        answer = current
        break
    seq.pop()
    seq.append("?")
    seq_text = ", ".join(str(x) for x in seq)
    return {
        "task_id": f"PAT-MIXED-{pat_difficulty}-{op_len}-{seq_len}-{k_max}",
        "skill": "pattern",
        "difficulty": pat_difficulty,
        "question": seq_text,
        "answer": answer,
        "answer_type": "integer",
        "mixed_op_len": op_len,
        "mixed_seq_len": seq_len
    }
    
def generate_mental_division(num_diff):
    spec = division_params_from_difficulty(num_diff)
    base = random.choice(spec["base_divisors"])
    answer = random.randint(*spec["answer_range"])
    steps = spec["simplify_steps"]
    divisor = base * 10
    dividend = divisor * answer
    question = f"What is {dividend} divided by {divisor}?"
    return {
        "task_id": f"NUM-DIV-{num_diff}",
        "skill": "numerical",
        "difficulty": num_diff,
        "question": question,
        "answer": answer,
        "answer_type": "integer",
        "division_base": base,
        "division_steps": steps
    }
    
def generate_ratio_scaling(num_diff):
    spec = ratio_params_from_difficulty(num_diff)
    k = spec["k"]
    a= random.randint(*spec["a_range"])
    mode = spec["mode"]
    if mode == "multiply":
        b = a * k
        while True:
            new_a = random.randint(*spec["a_range"])
            if a == new_a:
                continue
            else:
                break
        answer = new_a * k
        question = f"{a} -> {b}, {new_a} -> ?"
    elif mode == "divide":
        b = a * k
        while True:
            new_a = random.randint(*spec["a_range"])
            if a == new_a:
                continue
            else:
                break
        new_b = new_a * k
        answer = new_a
        question = f"{b} -> {a}, {new_b} -> ?"
    if mode == "multiply":
        task_id = f"NUM-RATIO-MUL-{num_diff}"
    else:
        task_id = f"NUM-RATIO-DIV-{num_diff}"
    return {
        "task_id": task_id,
        "skill": "numerical",
        "difficulty": num_diff,
        "question": question,
        "answer": answer,
        "answer_type": "integer",
        "ratio_k": k,
        "ratio_mode": mode
    }
    
def generate_square_task(num_diff):
    spec = square_params_from_difficulty(num_diff)
    base = random.randint(*spec["base_range"])
    offset = random.choice(spec["offsets"])
    if offset != 0:
        sign = random.choice([-1, 1])
        n = base + sign * offset
    else: 
        n = base
    answer = n*n
    question = f"What is {n}^2?"
    return {
        "task_id": f"NUM-SQUARE-{num_diff}",
        "skill": "numerical",
        "difficulty": num_diff,
        "question": question,
        "answer": answer,
        "answer_type": "integer",
        "square_base": base,
        "square_offset": offset,
        "square_mode": spec["mode"]
    }    

def generate_near_boundary_task(num_diff):
    boundary, offset, ops = near_boundary_param_from_difficulty(num_diff)
    boundary_value = random.choice(boundary)
    op = random.choice(ops)
    if num_diff >= 15:
        delta = random.randint(1, offset)
        sign = random.choice([-1, 1])
        a = boundary_value + sign * delta
        b = random.randint(1, offset * 2)
    else:
        delta_small = random.randint(1, max(1, offset // 3))
        delta_large = random.randint(offset // 2, offset)
        
        if op == "+":
            a = boundary_value - delta_small
            b = delta_large
            if a + b <= boundary_value:
                b = boundary_value - a + 1
        else: 
            a = boundary_value + delta_large
            b = delta_small
            if a - b >= boundary_value:
                b = a - boundary_value + 1
    if op == "+":
        answer = a + b
        question = f"What is {a} + {b}?"
    else:
        answer = a - b
        question = f"What is {a} - {b}?"
    if abs(answer - boundary_value) > 2 * offset:
        return generate_near_boundary_task(num_diff)
    crossed = (
        (op == "+" and a < boundary_value and answer > boundary_value) or
        (op == "-" and a > boundary_value and answer < boundary_value)
    )
    return {
        "task_id": f"NUM-NEARBOUNDARY-{num_diff}",
        "skill": "numerical",
        "difficulty": num_diff,
        "question": question,
        "answer": answer,
        "answer_type": "integer",
        "near_boundary_anchor": boundary_value,
        "near_boundary_op": op,
        "crossed_boundary": crossed
    }