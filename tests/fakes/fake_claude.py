"""Fake Claude CLI speaking the verified stream-json stdio protocol (no real claude spawned)."""
import json, os, sys, signal, time, subprocess, threading

MODE = sys.argv[1] if len(sys.argv) > 1 else 'normal'
SID = 'sess-123'  # echoes --session-id / --resume when given, like the real CLI does
for _i, _a in enumerate(sys.argv[:-1]):
    if _a in ('--session-id', '--resume'):
        SID = sys.argv[_i + 1]
NEVER = os.environ.get('FAKE_NEVER') == '1'  # long turn: never emits a result
TOOL = os.environ.get('FAKE_TOOL', 'Bash')  # tool named in the can_use_tool ask
TOOL_INPUT = json.loads(os.environ.get('FAKE_INPUT') or '{"command": "echo hi"}')
NOWAIT = os.environ.get('FAKE_NOWAIT') == '1'  # emit the ask, then finish at once (ask left pending)
CANCEL = os.environ.get('FAKE_CANCEL') == '1'  # emit the ask, withdraw it, then finish
SUGGESTIONS = json.loads(os.environ.get('FAKE_SUGGESTIONS') or 'null')  # permission_suggestions of the ask
ECHO = os.environ.get('FAKE_ECHO') == '1'  # result also carries the host's answer fields
NOCONV = os.environ.get('FAKE_NOCONV') == '1'  # --resume of a never-persisted uuid, as the real CLI does it
RESUMING = '--resume' in sys.argv
SLASH = json.loads(os.environ['FAKE_SLASH']) if os.environ.get('FAKE_SLASH') else None  # init slash_commands
FEED = os.environ.get('FAKE_FEED') == '1'  # progress-feed events before the permission ask (p3a tests)
TEXT = os.environ.get('FAKE_TEXT')  # NOWAIT only: an assistant message with this text, and the result carries it
USAGE = os.environ.get('FAKE_USAGE') == '1'  # NOWAIT only: the result carries total_cost_usd and usage (in 100, out 40)
CACHE = os.environ.get('FAKE_CACHE') == '1'  # NOWAIT only: usage with 12 fresh input and 80000 cache reads, no total_cost_usd
BG = os.environ.get('FAKE_BG')  # the first user message starts background work: bash | monitor | agent | agent_str | wakeup
BG_FLAG = os.environ.get('FAKE_BG_FLAG')  # the background work ends when this path exists ...
BG_SECONDS = float(os.environ.get('FAKE_BG_SECONDS') or '0.5')  # ... or after this many seconds
BG_QUICK = os.environ.get('FAKE_BG_QUICK') == '1'  # the end markers come before the first result, the self turn right after
BG_ASK = os.environ.get('FAKE_BG_ASK') == '1'  # while idle, a permission ask (perm_bg) comes before the end markers
PID_ANSWER = os.environ.get('FAKE_PID_ANSWER') == '1'  # every result text ends with ' pid=<this process id>'
OUT_LOCK = threading.Lock()  # the main thread and the background thread both write stdout
BG_ANSWER = threading.Event()  # set when the host answers perm_bg
BG_STATE = {'users': 0, 'worker': None}  # user messages seen; the background thread, when one runs
CALL_ID = 'toolu_bg'
WAIT_ANSWER = 30.0  # seconds the background thread waits for the answer to its ask
WAKE_MS = int(time.time() * 1000) + 500  # scheduledFor of a ScheduleWakeup (epoch ms)

def emit(obj):
    with OUT_LOCK:
        sys.stdout.write(json.dumps(obj) + '\n')
        sys.stdout.flush()

def tagged(text):
    return f'{text} pid={os.getpid()}' if PID_ANSWER else text

def result_line(text):
    return {'type': 'result', 'subtype': 'success', 'is_error': False, 'result': tagged(text), 'session_id': SID}

def bg_call():
    """The background tool call of FAKE_BG: (tool, input, start result, task id, task type). A wakeup has no task."""
    if BG == 'bash':
        return ('Bash', {'command': 'sleep 60', 'run_in_background': True}, {'backgroundTaskId': 'bash_task_1'},
                'bash_task_1', 'local_bash')
    if BG == 'monitor':
        return ('Monitor', {'command': 'tail -f log', 'description': 'watch'},
                {'taskId': 'mon_task_1', 'timeoutMs': 300000, 'persistent': False}, 'mon_task_1', 'local_bash')
    if BG in ('agent', 'agent_str'):
        flag = True if BG == 'agent' else 'true'
        agent = 'agent_1' if BG == 'agent' else 'agent_2'
        return ('Agent', {'description': 'bg', 'prompt': 'work', 'run_in_background': flag},
                {'isAsync': True, 'status': 'async_launched', 'agentId': agent}, agent, 'local_agent')
    return ('ScheduleWakeup', {'delaySeconds': 1, 'reason': 'check'}, {'scheduledFor': WAKE_MS}, None, None)

def bg_start():
    """FAKE_BG, first user message: the call starts the background work and the turn ends with STARTED."""
    name, tool_input, start, task_id, task_type = bg_call()
    emit({'type': 'system', 'subtype': 'init', 'session_id': SID, 'model': 'fake'})
    emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
          'message': {'content': [{'type': 'tool_use', 'id': CALL_ID, 'name': name, 'input': tool_input}]}})
    if task_id:
        emit({'type': 'system', 'subtype': 'background_tasks_changed',
              'tasks': [{'task_id': task_id, 'task_type': task_type, 'tool_use_id': CALL_ID}]})
        emit({'type': 'system', 'subtype': 'task_started', 'task_id': task_id, 'tool_use_id': CALL_ID,
              'is_backgrounded': True, 'task_type': task_type})
    emit({'type': 'user', 'session_id': SID,
          'message': {'content': [{'type': 'tool_result', 'tool_use_id': CALL_ID, 'content': 'started'}]},
          'tool_use_result': start})
    emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
          'message': {'content': [{'type': 'text', 'text': 'STARTED'}]}})
    if BG_QUICK:
        bg_end()
        emit(result_line('STARTED'))
        bg_self_turn()
        return
    emit(result_line('STARTED'))
    worker = threading.Thread(target=bg_later, daemon=True)
    BG_STATE['worker'] = worker
    worker.start()

def bg_later():
    """The background thread: waits for the flag or the seconds, asks when FAKE_BG_ASK, then ends the work and runs the
    self turn."""
    deadline = time.monotonic() + BG_SECONDS
    while not (BG_FLAG and os.path.exists(BG_FLAG)) and time.monotonic() < deadline:
        time.sleep(0.05)
    if BG_ASK:
        emit({'type': 'control_request', 'request_id': 'perm_bg',
              'request': {'subtype': 'can_use_tool', 'tool_name': 'Bash', 'input': {'command': 'date'}}})
        BG_ANSWER.wait(WAIT_ANSWER)
    bg_end()
    bg_self_turn()

def bg_end():
    """The end of the background work: a subagent line for an Agent, then the end markers (a wakeup: its lifecycle)."""
    _, _, _, task_id, _ = bg_call()
    if BG == 'wakeup':
        emit({'type': 'command_lifecycle', 'state': 'started', 'wakeup_source': 'loop_wakeup', 'scheduled_for': WAKE_MS})
        return
    if BG in ('agent', 'agent_str'):
        emit({'type': 'assistant', 'parent_tool_use_id': CALL_ID, 'session_id': SID, 'agent_id': task_id,
              'message': {'content': [{'type': 'text', 'text': 'AGENTOK'}]}})
    emit({'type': 'system', 'subtype': 'task_updated', 'task_id': task_id, 'patch': {'status': 'completed'}})
    emit({'type': 'system', 'subtype': 'task_notification', 'task_id': task_id, 'tool_use_id': CALL_ID,
          'status': 'completed'})
    emit({'type': 'system', 'subtype': 'background_tasks_changed', 'tasks': []})

def bg_self_turn():
    """The turn the CLI starts by itself: init, text, result. A task turn's result has the task-notification origin; a
    wakeup turn is wrapped in its command_lifecycle start (already sent) and completed lines, with no origin."""
    emit({'type': 'system', 'subtype': 'init', 'session_id': SID, 'model': 'fake'})
    if BG == 'wakeup':
        emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
              'message': {'content': [{'type': 'text', 'text': 'AWAKE'}]}})
        emit(result_line('AWAKE'))
        emit({'type': 'command_lifecycle', 'state': 'completed'})
        return
    emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
          'message': {'content': [{'type': 'text', 'text': 'FINISHED'}]}})
    final = result_line('FINISHED')
    final['origin'] = {'kind': 'task-notification'}
    emit(final)

def bg_user(msg):
    """FAKE_BG, a user message: the first starts the background work, later ones are answered as a plain turn."""
    BG_STATE['users'] += 1
    if BG_STATE['users'] == 1:
        bg_start()
        return
    emit({'type': 'system', 'subtype': 'init', 'session_id': SID, 'model': 'fake'})
    emit(result_line(f"ok:{BG_STATE['users']}"))

def read_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)

if MODE == 'stubborn':
    # ignores stdin EOF, interrupt, SIGINT and SIGTERM; spawns a child that inherits the group
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen(['sleep', '300'])
    sys.stderr.write('stubborn-stderr\n'); sys.stderr.flush()
    while True:
        time.sleep(1)

sys.stderr.write('fake-stderr-line\n'); sys.stderr.flush()
if os.environ.get('FAKE_STRAGGLER') == '1':  # TERM-ignoring child that stays in the session group
    subprocess.Popen(['bash', '-c', 'trap "" TERM; sleep 999'])
if NOCONV and RESUMING:
    # real CLI: no answer to initialize; one error result with no text in `errors`, the
    # message goes to stderr only, exit 1
    emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True, 'num_turns': 0,
          'result': f'No conversation found with session ID: {SID}', 'session_id': SID})
    sys.stderr.write(f'No conversation found with session ID: {SID}\n'); sys.stderr.flush()
    sys.exit(1)
if '--max-turns' in sys.argv:  # one-shot title call: `claude -p <prompt> --max-turns 1` prints one answer and exits
    if os.environ.get('FAKE_ONESHOT') == 'hang':
        g = subprocess.Popen(['sleep', '300'])  # a grandchild in the same process group, killed with the group
        if os.environ.get('FAKE_PIDFILE'):
            with open(os.environ['FAKE_PIDFILE'], 'w') as pf:
                pf.write(str(g.pid))
        time.sleep(300)
    sys.stdout.write(os.environ.get('FAKE_TITLE', 'Fake title for the topic') + '\n')
    sys.stdout.flush()
    sys.exit(0)
while True:
    msg = read_msg()
    if msg is None:  # stdin EOF -> clean exit, after the background thread has finished its self turn
        if BG_STATE['worker'] is not None:
            BG_STATE['worker'].join(WAIT_ANSWER + BG_SECONDS)
        sys.exit(0)
    if msg.get('type') == 'control_request':
        sub = msg['request']['subtype']
        if sub == 'initialize':
            emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': msg['request_id'], 'response': {'commands': []}}})
        elif sub in ('set_permission_mode', 'set_model'):
            emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': msg['request_id'], 'response': {}}})
        elif sub == 'interrupt':
            emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': msg['request_id'], 'response': {'still_queued': []}}})
            # the real CLI always answers an interrupt; in BG mode stop() also interrupts while idle, and a stray result
            # there would look like the answer to a turn, so BG mode sends none
            if (not NEVER or os.environ.get('FAKE_INTERRUPT_RESULT') == '1') and not BG:
                emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True, 'result': None, 'session_id': SID})
    elif BG and msg.get('type') == 'control_response':
        BG_ANSWER.set()  # the host answered the idle ask of FAKE_BG_ASK
    elif BG and msg.get('type') == 'user':
        bg_user(msg)
    elif msg.get('type') == 'user':
        init = {'type': 'system', 'subtype': 'init', 'session_id': SID, 'model': 'fake'}
        if SLASH is not None:  # p3c: the commands the real CLI lists in init
            init['slash_commands'] = SLASH
        emit(init)
        if FEED:
            emit({'type': 'system', 'subtype': 'thinking_tokens', 'session_id': SID, 'tokens': 5})
            emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
                  'message': {'content': [{'type': 'text', 'text': 'Let me check.'}]}})
            emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
                  'message': {'content': [{'type': 'tool_use', 'id': 'tu_1', 'name': 'Read', 'input': {'file_path': os.path.join(os.getcwd(), 'x.txt')}}]}})
            emit({'type': 'user', 'parent_tool_use_id': None, 'session_id': SID,
                  'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'tu_1', 'is_error': True, 'content': 'File does not exist.\nmore'}]}})
            emit({'type': 'system', 'subtype': 'api_retry', 'session_id': SID, 'attempt': 1, 'max_retries': 10, 'error': 'rate_limit', 'retry_delay_ms': 8000})
            emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
                  'message': {'content': [{'type': 'tool_use', 'id': 'tu_2', 'name': 'Bash', 'input': {'command': 'ls\n-la', 'description': 'List files'}}]}})
            emit({'type': 'assistant', 'parent_tool_use_id': 'tu_2', 'session_id': SID,
                  'message': {'content': [{'type': 'tool_use', 'id': 'tu_3', 'name': 'Glob', 'input': {'pattern': '**/*.py'}}]}})
            emit({'type': 'assistant', 'parent_tool_use_id': 'tu_2', 'session_id': SID,
                  'message': {'content': [{'type': 'text', 'text': 'SUBAGENT-SECRET'}]}})
            emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
                  'message': {'content': [{'type': 'text', 'text': 'ok:allow'}]}})
        if os.environ.get('FAKE_GARBAGE') == '1':  # binary junk and a broken JSON-looking line
            sys.stdout.flush()
            sys.stdout.buffer.write(b'\xff\xfe\x80 junk\n{\xff broken\n')
            sys.stdout.buffer.flush()
        if os.environ.get('FAKE_SUBAGENT') == '1':  # subagent text (parent set) and the main answer text
            emit({'type': 'assistant', 'parent_tool_use_id': 'toolu_sub', 'session_id': SID,
                  'message': {'content': [{'type': 'text', 'text': 'SUBAGENT-SECRET'}]}})
            emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
                  'message': {'content': [{'type': 'text', 'text': 'MAIN-ANSWER'}]}})
        if os.environ.get('FAKE_DIE') == '1':  # exit without a result; a grandchild keeps stdout open
            g = subprocess.Popen(['sleep', '30'], stdout=sys.stdout, start_new_session=True)
            with open(os.environ['FAKE_DIE_PIDFILE'], 'w') as pf:
                pf.write(str(g.pid))
            sys.stderr.write('fake-die-stderr\n'); sys.stderr.flush()
            sys.exit(3)
        if NEVER:
            continue  # keep reading: only interrupt / EOF end this turn
        emit({'type': 'control_request', 'request_id': 'perm_1', 'request': {
            'subtype': 'can_use_tool', 'tool_name': TOOL, 'input': TOOL_INPUT,
            **({'permission_suggestions': SUGGESTIONS} if SUGGESTIONS is not None else {})}})
        if NOWAIT:
            if TEXT is not None:
                emit({'type': 'assistant', 'parent_tool_use_id': None, 'session_id': SID,
                      'message': {'content': [{'type': 'text', 'text': TEXT}]}})
            extra = {'total_cost_usd': 0.5, 'usage': {'input_tokens': 100, 'output_tokens': 40}} if USAGE else {}
            if CACHE:
                extra = {'usage': {'input_tokens': 12, 'cache_read_input_tokens': 80000, 'output_tokens': 40}}
            emit({'type': 'result', 'subtype': 'success', 'is_error': False,
                  'result': tagged(TEXT if TEXT is not None else 'ok:unanswered'), 'session_id': SID, **extra})
            continue
        if CANCEL:
            emit({'type': 'control_cancel_request', 'request_id': 'perm_1'})
            emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'ok:cancelled', 'session_id': SID})
            continue
        while True:  # wait for the host's permission answer
            r = read_msg()
            if r is None:
                sys.exit(0)
            if r.get('type') == 'control_request':  # host mode switch while an ask is open: answer it, keep waiting
                emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': r['request_id'], 'response': {}}})
                continue
            if r.get('type') == 'control_response' and r['response'].get('request_id') == 'perm_1':
                resp = r['response']['response']
                behavior = resp['behavior']
                echo = {k: resp[k] for k in ('updatedInput', 'updatedPermissions', 'message') if k in resp}
                break
        emit({'type': 'result', 'subtype': 'success', 'is_error': False,
              'result': tagged(f'ok:{behavior}' + (' ' + json.dumps(echo, ensure_ascii=False, sort_keys=True) if ECHO else '')),
              'session_id': SID})
