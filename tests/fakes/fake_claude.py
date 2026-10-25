"""Fake Claude CLI speaking the verified stream-json stdio protocol (no real claude spawned)."""
import json, os, sys, signal, time, subprocess

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

def emit(obj):
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()

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
    if msg is None:  # stdin EOF -> clean exit
        sys.exit(0)
    if msg.get('type') == 'control_request':
        sub = msg['request']['subtype']
        if sub == 'initialize':
            emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': msg['request_id'], 'response': {'commands': []}}})
        elif sub in ('set_permission_mode', 'set_model'):
            emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': msg['request_id'], 'response': {}}})
        elif sub == 'interrupt':
            emit({'type': 'control_response', 'response': {'subtype': 'success', 'request_id': msg['request_id'], 'response': {'still_queued': []}}})
            if not NEVER or os.environ.get('FAKE_INTERRUPT_RESULT') == '1':  # the real CLI always answers an interrupt
                emit({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True, 'result': None, 'session_id': SID})
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
                  'result': TEXT if TEXT is not None else 'ok:unanswered', 'session_id': SID, **extra})
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
              'result': f'ok:{behavior}' + (' ' + json.dumps(echo, ensure_ascii=False, sort_keys=True) if ECHO else ''),
              'session_id': SID})
