# Managed terminal configuration. Credentials never belong in this file.
typeset -U path PATH
path=("$HOME/.local/bin" $path)

proxy_on() {
    local settings host port
    settings="$(/usr/sbin/scutil --proxy)"
    host="$(print -r -- "$settings" | /usr/bin/awk '/HTTPProxy :/ {print $3; exit}')"
    port="$(print -r -- "$settings" | /usr/bin/awk '/HTTPPort :/ {print $3; exit}')"
    if [[ -n "$host" && "$port" == <-> ]]; then
        export HTTP_PROXY="http://${host}:${port}"
    else
        export HTTP_PROXY="${TERMINAL_PROXY_URL:-http://127.0.0.1:25662}"
    fi
    export HTTPS_PROXY="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY"
    export http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
    export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
    export no_proxy="$NO_PROXY"
}

proxy_off() {
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
}

proxy_status() {
    print -r -- "HTTPS_PROXY=${HTTPS_PROXY:-off}"
    command curl --silent --show-error --fail --connect-timeout 5 --max-time 15 \
        https://api.github.com/ -o /dev/null && print 'GitHub connection OK'
}

proxy_on
