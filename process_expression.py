def split_args(args_str):
    args = []
    current = []
    depth = 0
    for c in args_str:
        if c == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            current.append(c)
    if current:
        args.append(''.join(current).strip())
    return args

def parse_function_call(expr):
    expr = expr.strip()
    if '(' not in expr or not expr.endswith(')'):
        return None, [expr]
    paren_open = expr.find('(')
    func_name = expr[:paren_open].strip()
    args_str = expr[paren_open+1:-1].strip()
    return func_name, split_args(args_str)

def replace_var_name(var_name, delay):
    if "_mrq_" not in var_name:
        return var_name
    if delay == 0:
        return var_name
    parts = var_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
    else:
        base = var_name
    return f"{base}_{delay}"

def process_expression(expr, current_delay=0, max_dely=12):
    func_name, args = parse_function_call(expr)
    if func_name is None:
        return replace_var_name(expr, current_delay)
    elif func_name.startswith('SUE'):
        # 处理 SUE_2(sale_mrq_1) 形式的调用
        if '_' in func_name:
            func_parts = func_name.split('_')
            if len(func_parts) == 2 and func_parts[0] == 'SUE' and func_parts[1].isdigit():
                periods = int(func_parts[1])
                if len(args) != 1:
                    raise ValueError(f"SUE_{periods} requires exactly 1 argument, got {len(args)}")
                var_expr = args[0]
                # 如果变量名不包含 _mrq_，直接返回原变量
                if "_mrq_" not in var_expr:
                    return var_expr
                delay_calls = []
                for i in range(periods):
                    delay_expr = f"Findelay({var_expr}, {i})"
                    processed_delay = process_expression(delay_expr, current_delay, max_dely)
                    delay_calls.append(processed_delay)
                # 生成 mean(data) 和 std(data)
                data = ', '.join(delay_calls)
                mean_expr = f"mean({data})"
                std_expr = f"std({data})"
                # 生成 sub(data[0], mean(data)) 并包裹在 truediv 中
                return f"truediv(sub({delay_calls[0]}, {mean_expr}), {std_expr})"
        # 处理 SUE(sale_mrq_1, periods) 形式的调用
        if len(args) != 2:
            raise ValueError(f"SUE requires exactly 2 arguments, got {len(args)}")
        var_expr, periods_expr = args[0], args[1]
        processed_periods = process_expression(periods_expr, current_delay)
        try:
            periods_num = int(processed_periods)
        except ValueError:
            raise ValueError(f"SUE's second argument must be an integer, got '{processed_periods}'")
        if periods_num <= 0:
            raise ValueError(f"SUE's periods must be positive, got {periods_num}")
        # 如果变量名不包含 _mrq_，直接返回原变量
        if "_mrq_" not in var_expr:
            return var_expr
        delay_calls = []
        for i in range(periods_num):
            delay_expr = f"Findelay({var_expr}, {i})"
            processed_delay = process_expression(delay_expr, current_delay, max_dely)
            delay_calls.append(processed_delay)
        # 生成 mean(data) 和 std(data)
        data = ', '.join(delay_calls)
        mean_expr = f"mean({data})"
        std_expr = f"std({data})"
        # 生成 sub(data[0], mean(data)) 并包裹在 truediv 中
        return f"truediv(sub({delay_calls[0]}, {mean_expr}), {std_expr})"
    elif func_name.startswith('Findelay'):
        if func_name == 'Findelay':
            if len(args) != 2:
                raise ValueError(f"Invalid Findelay: {expr}")
            inner_expr, delay_str = args[0], args[1]
        else:
            delay_prefix = func_name.split('_', 1)
            if len(delay_prefix) != 2 or not delay_prefix[1].isdigit():
                raise ValueError(f"Invalid Findelay function: {func_name}")
            delay_str = delay_prefix[1]
            inner_expr = args[0]
        try:
            additional_delay = int(delay_str)
        except ValueError:
            raise ValueError(f"Invalid Findelay value: {delay_str}")
        new_delay = current_delay + additional_delay
        if new_delay > max_dely:
            raise ValueError(f"max_dely is {max_dely}, Invalid Findelay: {new_delay} > {max_dely}")
        return process_expression(inner_expr, new_delay, max_dely)
    else:
        new_args = [process_expression(arg, current_delay, max_dely) for arg in args]
        return f"{func_name}({', '.join(new_args)})"