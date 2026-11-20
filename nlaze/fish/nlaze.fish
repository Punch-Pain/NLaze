# NLaze v6 fish shell integration (spec §1, §2, §4, §5, §10)
# Source this file in config.fish or via `nlaze fish-init | source`.
# Kill switch: `set -g NLAZE_DISABLE 1` disables the hook immediately.

function __nlaze_postexec --on-event fish_postexec
    set -l last_status $status
    if set -q NLAZE_DISABLE; and test "$NLAZE_DISABLE" = "1"
        return
    end
    set -l cmd_line "$argv"
    if test -z "$cmd_line"
        return
    end

    set -l nlaze_bin (set -q NLAZE_BIN; and echo "$NLAZE_BIN"; or echo "nlaze")
    set -l payload ($nlaze_bin --json hook-postexec --status $last_status --cwd "$PWD" -- "$cmd_line" 2>/dev/null)
    if test -z "$payload"
        return
    end

    set -l action (printf '%s' "$payload" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("action","passthrough"))' 2>/dev/null)
    if test "$action" = "passthrough" -o -z "$action"
        return
    end

    set -l corr (printf '%s' "$payload" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("correction",""))' 2>/dev/null)
    set -l conf (printf '%s' "$payload" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(int(round(float(d.get("confidence",0))*100)))' 2>/dev/null)
    set -l head (printf '%s' "$payload" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("head",""))' 2>/dev/null)
    set -l flag (printf '%s' "$payload" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("flag",""))' 2>/dev/null)

    switch "$action"
        case auto
            echo "[nlaze auto] $corr"
            eval -- "$corr"

        case certify_head
            echo "NLaze wants to auto-correct to: $corr"
            echo "Model confidence: $conf%. This is a one-time request."
            read -l -P "Allow auto-correcting to '$head'? [yes/no] " ans
            if test "$ans" = "yes" -o "$ans" = "y"
                $nlaze_bin feedback --input "$cmd_line" --correction "$corr" --accept --certify head --head "$head" >/dev/null 2>&1
                eval -- "$corr"
            else
                $nlaze_bin feedback --input "$cmd_line" --correction "$corr" --reject --certify head --head "$head" >/dev/null 2>&1
            end

        case certify_flag
            echo "NLaze wants to auto-correct to: $corr"
            echo "The command '$head' is allowed, but flag '$flag' may be dangerous."
            read -l -P "Allow this flag? [yes/no] " ans
            if test "$ans" = "yes" -o "$ans" = "y"
                $nlaze_bin feedback --input "$cmd_line" --correction "$corr" --accept --certify flag --head "$head" --flag "$flag" >/dev/null 2>&1
                eval -- "$corr"
            else
                $nlaze_bin feedback --input "$cmd_line" --correction "$corr" --reject --certify flag --head "$head" --flag "$flag" >/dev/null 2>&1
            end

        case confirm confirm_mild
            read -l -P "[nlaze] Did you mean: $corr ? [y/N] " ans
            if test "$ans" = "y" -o "$ans" = "Y" -o "$ans" = "yes"
                $nlaze_bin feedback --input "$cmd_line" --correction "$corr" --accept >/dev/null 2>&1
                eval -- "$corr"
            else
                $nlaze_bin feedback --input "$cmd_line" --correction "$corr" --reject >/dev/null 2>&1
            end

        case confirm_high
            echo "[nlaze HIGH DANGER] Proposed correction is destructive/privileged:"
            echo "  $corr"
            read -l -P "Type 'yes' to execute (any other input cancels): " ans
            if test "$ans" = "yes"
                eval -- "$corr"
            end
    end
end
