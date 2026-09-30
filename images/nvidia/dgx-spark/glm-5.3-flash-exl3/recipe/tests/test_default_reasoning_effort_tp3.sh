#!/usr/bin/env bash
# GLM53_DEFAULT_REASONING_EFFORT on the TP=3 launcher.
#
# The knob was declared, guarded and forwarded on start.sh (TP=2) only, so a
# TP3 seat silently ignored it: no declaration, no enum guard, no
# --default-chat-template-kwargs in either inner script, nothing in serve_env
# or the head's -e list. An operator setting it in the shared .env got a no-op
# and every client that sent no reasoning_effort fell through to
# files/chat_template.jinja, which resolves an absent effort to `max`.
#
# The same five claims tests/test_default_reasoning_effort.sh makes for start.sh
# are made here against start-tp3.sh's own text, plus the two places TP3 keeps
# its per-rank env (the shared serve_env loop for both workers, and the head's
# explicit -e list). Values are passed through the environment so adversarial
# strings are exercised safely rather than interpolated into source.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
START="$HERE/../start-tp3.sh"
[ -f "$START" ] || { echo "start-tp3.sh not found" >&2; exit 1; }
fail=0
W="$$_tp3effort"

# ---------------------------------------------------------------- guard ----
# Same marker block the TP2 test slices. start-tp3.sh's validate_numeric_config
# reads more variables than the TP2 one, so the probe below stands the whole
# guard up with the launcher's own defaults and varies only the knob.
guard="$(sed -n '/^# GLM53 numeric config guard (begin)$/,/^# GLM53 numeric config guard (end)$/p' "$START")"
[ -n "$guard" ] || { echo "numeric config guard block not found" >&2; exit 1; }
printf '%s\n' "$guard" > "/tmp/_effort_tp3_guard.$W"

# validate_numeric_config exits non-zero on any refusal; a probe that returns 2
# for an unrelated missing variable would make every row below meaningless, so
# the unsetting baseline is asserted first and must pass.
guard_rc() { # echoes validate_numeric_config's rc for knob value $1
    GLM53_DEFAULT_REASONING_EFFORT="$1" bash -c '
        source "/tmp/_effort_tp3_guard.'"$W"'"
        GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
        GLM53_SPINWAIT_MS=stock GLM53_HOST_MEM_HYGIENE=0
        GLM53_KDA_BF16_LARGE_M=0 GLM53_DENSE_EXL3=0 HAREM_KDA_FLASHKDA=0
        GLM53_COOP_GEOMETRY="" GLM53_ADAPTIVE_K_SET="" GLM53_DRAFT_KV_COMPACT=0
        SPEC_METHOD=none MTP_TOKENS=0 FLASHKDA_PATCH_HOST=/nonexistent
        validate_numeric_config' >/dev/null 2>&1
    echo $?
}

if [ "$(guard_rc low)" = "0" ]; then
    echo "ok   guard baseline (low) -> rc 0"
else
    echo "FAIL guard baseline does not reach rc 0; probe env is incomplete"; fail=1
fi

check_guard() {
    local got; got="$(guard_rc "$1")"
    if [ "$got" = "$2" ]; then echo "ok   guard [$1] -> rc $got"
    else echo "FAIL guard [$1] -> rc $got want $2"; fail=1; fi
}
check_guard ""       0
check_guard "low"    0
check_guard "high"   0
check_guard "max"    0
check_guard "medium" 2
check_guard "High"   2
check_guard "MAX"    2
check_guard " high"  2
check_guard "high "  2
check_guard "junk"   2
check_guard 'high;id' 2

# an UNSET knob must pass too (the guard reads ${VAR-}, not $VAR under set -u)
if bash -c 'set -u; source "/tmp/_effort_tp3_guard.'"$W"'"
    GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
    GLM53_SPINWAIT_MS=stock GLM53_HOST_MEM_HYGIENE=0
    GLM53_KDA_BF16_LARGE_M=0 GLM53_DENSE_EXL3=0 HAREM_KDA_FLASHKDA=0
    GLM53_COOP_GEOMETRY="" GLM53_ADAPTIVE_K_SET="" GLM53_DRAFT_KV_COMPACT=0
    SPEC_METHOD=none MTP_TOKENS=0 FLASHKDA_PATCH_HOST=/nonexistent
    validate_numeric_config' >/dev/null 2>&1; then
    echo "ok   guard [<unset>] -> rc 0"
else
    echo "FAIL guard [<unset>] must pass with the var unset"; fail=1
fi

# the rejection must name the knob and the three legal values
msg="$(GLM53_DEFAULT_REASONING_EFFORT=medium bash -c '
    source "/tmp/_effort_tp3_guard.'"$W"'"
    GPU_MEM_UTIL=0.87 MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048
    GLM53_SPINWAIT_MS=stock GLM53_HOST_MEM_HYGIENE=0
    GLM53_KDA_BF16_LARGE_M=0 GLM53_DENSE_EXL3=0 HAREM_KDA_FLASHKDA=0
    GLM53_COOP_GEOMETRY="" GLM53_ADAPTIVE_K_SET="" GLM53_DRAFT_KV_COMPACT=0
    SPEC_METHOD=none MTP_TOKENS=0 FLASHKDA_PATCH_HOST=/nonexistent
    validate_numeric_config' 2>&1 >/dev/null || true)"
case "$msg" in
    *GLM53_DEFAULT_REASONING_EFFORT*low*high*max*) echo "ok   guard error names knob and enum" ;;
    *) echo "FAIL guard error text: [$msg]"; fail=1 ;;
esac

# ----------------------------------------------------------- serve args ----
# Slice each inner script out of its quoted heredoc, then keep only the ARGS
# construction (ARGS=( ... up to the config.json existence check).
args_block() { # $1 = HEAD_SCRIPT | WORKER_SCRIPT
    awk -v var="$1" '
        index($0, "cat > \"$" var "\" <<") { inblk = 1; next }
        inblk && $0 == "EOF" { exit }
        inblk { print }
    ' "$START" | sed -n '/^ARGS=(/,/^\[ -f "${MODEL_DIR}\/config.json"/p' | sed '$d'
}

for var in HEAD_SCRIPT WORKER_SCRIPT; do
    args_block "$var" > "/tmp/_effort_tp3_args_${var}.$W"
    if ! [ -s "/tmp/_effort_tp3_args_${var}.$W" ]; then
        echo "FAIL could not slice ARGS block for $var"; fail=1; continue
    fi
    if ! grep -q -- '--default-chat-template-kwargs' "/tmp/_effort_tp3_args_${var}.$W"; then
        echo "FAIL $var ARGS block has no --default-chat-template-kwargs"; fail=1
    fi
done

emit() { # $1 = HEAD_SCRIPT|WORKER_SCRIPT, $2 = knob value; argv one per line
    GLM53_DEFAULT_REASONING_EFFORT="$2" bash -c '
        say() { :; }
        SERVED_MODEL_NAME=m PORT=8888 TP=3 NNODES=3 HEAD_IP=10.0.0.1 MASTER_PORT=29500
        ENFORCE_EAGER=0 QUANTIZATION=none MAX_MODEL_LEN=1000000 GPU_MEM_UTIL=0.87
        MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 KV_CACHE_DTYPE=fp8
        SPEC_METHOD=none MTP_TOKENS=0 CHAT_TEMPLATE= LANGUAGE_MODEL_ONLY=0
        LIMIT_MM= SKIP_MM_PROFILING=0 EXTRA_ARGS=
        source "/tmp/_effort_tp3_args_'"$1"'.'"$W"'"
        printf "%s\n" "${ARGS[@]}"' 2>/dev/null
}

check_emit() { # $1 rank var, $2 value, $3 expected JSON ("" = flag must be absent)
    local out n json
    out="$(emit "$1" "$2")"
    n="$(printf '%s\n' "$out" | grep -c -- '^--default-chat-template-kwargs$' || true)"
    json="$(printf '%s\n' "$out" | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
    if [ -z "$3" ]; then
        if [ "$n" = "0" ]; then echo "ok   $1 [$2] -> no flag"
        else echo "FAIL $1 [$2] -> emitted $n flag(s): [$json]"; fail=1; fi
        return
    fi
    if [ "$n" != "1" ]; then
        echo "FAIL $1 [$2] -> flag emitted $n times (want exactly 1)"; fail=1; return
    fi
    if [ "$json" != "$3" ]; then
        echo "FAIL $1 [$2] -> JSON [$json] want [$3]"; fail=1; return
    fi
    case "$json" in
        *" "*) echo "FAIL $1 [$2] -> JSON contains a space: [$json]"; fail=1; return ;;
    esac
    echo "ok   $1 [$2] -> $json"
}

for var in HEAD_SCRIPT WORKER_SCRIPT; do
    check_emit "$var" ""     ""
    check_emit "$var" "low"  '{"reasoning_effort":"low"}'
    check_emit "$var" "high" '{"reasoning_effort":"high"}'
    check_emit "$var" "max"  '{"reasoning_effort":"max"}'
done

# both ranks must build the identical flag from the same knob
h="$(emit HEAD_SCRIPT high | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
w="$(emit WORKER_SCRIPT high | grep -A1 -- '^--default-chat-template-kwargs$' | sed -n '2p')"
if [ "$h" = "$w" ] && [ -n "$h" ]; then echo "ok   both ranks build the identical flag"
else echo "FAIL rank flags differ: head [$h] worker [$w]"; fail=1; fi

# -------------------------------------------------------- per-rank env ----
launcher="$(cat "$START")"

# Workers: TP3 keeps their env in one shared serve_env loop covering ranks 1
# and 2, so the knob has to be a member of that list.
if printf '%s\n' "$launcher" | sed -n '/for v in SERVED_MODEL_NAME/,/; do/p' \
     | grep -q 'GLM53_DEFAULT_REASONING_EFFORT'; then
    echo "ok   knob reaches both worker ranks via serve_env"
else
    echo "FAIL knob is missing from the serve_env loop"; fail=1
fi

# Head: its own explicit -e list. The trailing backslash is part of the claim:
# without it the docker command would end on this line and the next -e would
# be a separate command, which is exactly the kind of break that only shows up
# as a failed launch.
head_line="$(printf '%s\n' "$launcher" \
    | grep -- '-e "GLM53_DEFAULT_REASONING_EFFORT=' | head -1)"
case "$head_line" in
    *'\'*) echo "ok   head -e list carries the knob and continues the command" ;;
    "")    echo "FAIL knob is not exported into the head container"; fail=1 ;;
    *)     echo "FAIL head -e line has no line continuation: [$head_line]"; fail=1 ;;
esac

# default must stay empty: this changes no behaviour until an operator opts in
case "$launcher" in
    *'GLM53_DEFAULT_REASONING_EFFORT="${GLM53_DEFAULT_REASONING_EFFORT-}"'*)
        echo "ok   launcher default is empty (unchanged behaviour)" ;;
    *) echo "FAIL launcher default is not empty"; fail=1 ;;
esac

# .env.tp3 is sourced AFTER the declaration, so the TP3 env file still wins.
decl="$(printf '%s\n' "$launcher" | grep -n 'GLM53_DEFAULT_REASONING_EFFORT="${GLM53_DEFAULT_REASONING_EFFORT-}"' | head -1 | cut -d: -f1)"
src="$(printf '%s\n' "$launcher" | grep -n 'source "$SCRIPT_DIR/.env.tp3"' | head -1 | cut -d: -f1)"
if [ -n "$decl" ] && [ -n "$src" ] && [ "$decl" -lt "$src" ]; then
    echo "ok   declaration precedes the .env.tp3 source (env file still wins)"
else
    echo "FAIL declaration ($decl) must precede the .env.tp3 source ($src)"; fail=1
fi

rm -f "/tmp/_effort_tp3_guard.$W" "/tmp/_effort_tp3_args_HEAD_SCRIPT.$W" \
      "/tmp/_effort_tp3_args_WORKER_SCRIPT.$W"
[ "$fail" = 0 ] && echo "default reasoning effort TP3 tests: PASS"
exit $fail
