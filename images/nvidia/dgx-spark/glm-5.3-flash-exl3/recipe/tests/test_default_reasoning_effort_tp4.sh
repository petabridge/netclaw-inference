#!/usr/bin/env bash
# GLM53_DEFAULT_REASONING_EFFORT on the TP=4 launcher.
#
# start.sh (TP=2) and start-tp3.sh (#299) wire this knob into
# --default-chat-template-kwargs; start-tp4.sh ignored it, so a value in the
# shared .env reached no rank and clients that sent no reasoning_effort fell
# through to files/chat_template.jinja, which resolves an absent effort to `max`.
#
# Runs start-tp4.sh's own guard and ARGS construction; values go through the
# environment so hostile strings are exercised rather than interpolated.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
START="$HERE/../start-tp4.sh"
[ -f "$START" ] || { echo "start-tp4.sh not found" >&2; exit 1; }
fail=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------- guard ----
sed -n '/^# GLM53 numeric config guard (begin)$/,/^# GLM53 numeric config guard (end)$/p' \
    "$START" > "$TMP/guard.sh"
[ -s "$TMP/guard.sh" ] || { echo "numeric config guard block not found" >&2; exit 1; }

PROBE_ENV='
    GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
    SPEC_METHOD=none DFLASH_TOKENS=7 GLM53_ADAPTIVE_K=off GLM53_SPINWAIT_MS=stock
    GLM53_DENSE_EXL3=0 GLM53_DRAFT_KV_COMPACT=0'

guard_rc() { # $1 = knob value, or the literal UNSET
    if [ "$1" = UNSET ]; then
        bash -c "set -u; unset GLM53_DEFAULT_REASONING_EFFORT; source '$TMP/guard.sh'; $PROBE_ENV
            validate_numeric_config" >/dev/null 2>&1
    else
        GLM53_DEFAULT_REASONING_EFFORT="$1" bash -c "source '$TMP/guard.sh'; $PROBE_ENV
            validate_numeric_config" >/dev/null 2>&1
    fi
    echo $?
}

# A baseline that fails for an unrelated reason would make every row below
# meaningless, so it must pass first.
if [ "$(guard_rc low)" != "0" ]; then
    echo "FAIL guard baseline (low) does not reach rc 0; probe env is incomplete"
    exit 1
fi
check_guard() {
    local got; got="$(guard_rc "$1")"
    if [ "$got" = "$2" ]; then echo "ok   guard [$1] -> rc $got"
    else echo "FAIL guard [$1] -> rc $got want $2"; fail=1; fi
}
check_guard UNSET    0
check_guard ""       0
check_guard low      0
check_guard high     0
check_guard max      0
check_guard medium   2
check_guard High     2
check_guard " high"  2
check_guard 'high;id' 2
check_guard '"}'     2

msg="$(GLM53_DEFAULT_REASONING_EFFORT=medium bash -c "source '$TMP/guard.sh'; $PROBE_ENV
    validate_numeric_config" 2>&1 >/dev/null || true)"
case "$msg" in
    *GLM53_DEFAULT_REASONING_EFFORT*low*high*max*) echo "ok   guard error names knob and values" ;;
    *) echo "FAIL guard error text: [$msg]"; fail=1 ;;
esac

# ----------------------------------------------------------- serve args ----
# Each rank's ARGS construction, sliced out of its quoted heredoc.
args_block() { # $1 = HEAD_SCRIPT | WORKER_SCRIPT
    awk -v var="$1" '
        index($0, "cat > \"$" var "\" <<") { inblk = 1; next }
        inblk && $0 == "EOF" { exit }
        inblk { print }
    ' "$START" | sed -n '/^ARGS=(/,/^\[ -f "${MODEL_DIR}\/config.json"/p' | sed '$d'
}
for var in HEAD_SCRIPT WORKER_SCRIPT; do
    args_block "$var" > "$TMP/args_$var.sh"
    [ -s "$TMP/args_$var.sh" ] || { echo "FAIL could not slice ARGS block for $var"; fail=1; }
done

emit() { # $1 = rank script, $2 = knob value; prints argv one per line
    GLM53_DEFAULT_REASONING_EFFORT="$2" bash -c "
        say() { :; }
        SERVED_MODEL_NAME=m PORT=8888 TP=4 NNODES=4 HEAD_IP=10.0.0.1 MASTER_PORT=29500
        ENFORCE_EAGER=0 QUANTIZATION=none MAX_MODEL_LEN=1000000 GPU_MEM_UTIL=0.87
        MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 KV_CACHE_DTYPE=fp8
        SPEC_METHOD=none MTP_TOKENS=0 CHAT_TEMPLATE= LANGUAGE_MODEL_ONLY=0
        LIMIT_MM= SKIP_MM_PROFILING=0 EXTRA_ARGS=
        source '$TMP/args_$1.sh'
        printf '%s\n' \"\${ARGS[@]}\"" 2>/dev/null
}
check_emit() { # $1 rank, $2 value, $3 expected JSON ("" = flag absent)
    local out n json
    out="$(emit "$1" "$2")"
    n="$(printf '%s\n' "$out" | grep -c -- '^--default-chat-template-kwargs$' || true)"
    json="$(printf '%s\n' "$out" | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
    if [ -z "$3" ]; then
        [ "$n" = 0 ] && echo "ok   $1 [$2] -> no flag" \
            || { echo "FAIL $1 [$2] -> flag emitted: [$json]"; fail=1; }
    elif [ "$n" = 1 ] && [ "$json" = "$3" ]; then
        echo "ok   $1 [$2] -> $json"
    else
        echo "FAIL $1 [$2] -> $n flag(s) [$json] want [$3]"; fail=1
    fi
}
for var in HEAD_SCRIPT WORKER_SCRIPT; do
    check_emit "$var" ""     ""
    check_emit "$var" low    '{"reasoning_effort":"low"}'
    check_emit "$var" high   '{"reasoning_effort":"high"}'
    check_emit "$var" max    '{"reasoning_effort":"max"}'
done

# -------------------------------------------------------- rank env ----
# nccl_common reaches rank 0 directly and ranks 1-3 through worker_nccl, so
# membership there is what puts the knob in all four containers.
if sed -n '/local -a nccl_common=(/,/^    )$/p' "$START" \
     | grep -qF -- '-e "GLM53_DEFAULT_REASONING_EFFORT=${GLM53_DEFAULT_REASONING_EFFORT-}"'; then
    echo "ok   knob is in nccl_common (all four ranks)"
else
    echo "FAIL knob is not forwarded through nccl_common"; fail=1
fi

# .env.tp4 must still win: the declaration comes before it is sourced.
decl="$(grep -n '^GLM53_DEFAULT_REASONING_EFFORT="${GLM53_DEFAULT_REASONING_EFFORT-}"$' "$START" | head -1 | cut -d: -f1)"
src="$(grep -n '^source "$SCRIPT_DIR/.env.tp4"$' "$START" | head -1 | cut -d: -f1)"
if [ -n "$decl" ] && [ -n "$src" ] && [ "$decl" -lt "$src" ]; then
    echo "ok   empty-default declaration precedes the .env.tp4 source"
else
    echo "FAIL declaration ($decl) must exist and precede the .env.tp4 source ($src)"; fail=1
fi

[ "$fail" = 0 ] && echo "default reasoning effort TP4 tests: PASS"
exit $fail
