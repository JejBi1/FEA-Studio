#!/bin/sh
# FEA Studio – instalace na Linuxu (x86_64, glibc 2.35+: Ubuntu 22.04+, Debian 12+, Fedora 36+).
#   ./install.sh [install|update] [--profile minimal|standard|full] [modul ...] [--no-shortcuts] [--no-check] [--force]
# update = po rozbalení nové verze přes starou (migrace, zastaralé soubory, moduly jako minule).
# Bez parametrů se zeptá na rozsah. Vše se instaluje do této složky (.venv, tools) bez práv správce;
# další moduly později: .venv/bin/python -m fea_core.modules install <modul>
# Podrobnosti: installer/README.md
set -e
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONIOENCODING=utf-8

# první parametr smí být příkaz instalátoru, jinak install
cmd=install
case "${1:-}" in
    install|update|remove|list|doctor|uninstall|shortcuts|slim) cmd="$1"; shift ;;
esac

py=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        py="$c"
        break
    fi
done

if [ -n "$py" ]; then
    exec "$py" installer/fea_setup.py "$cmd" "$@"
fi

# žádný Python 3.9+: uv (stažený do tools/uv) spustí instalátor svým Pythonem 3.11
echo "Python 3.9+ nenalezen – stahuji správce balíčků uv (Python 3.11 si stáhne sám)."
command -v curl >/dev/null 2>&1 || { echo "Chybí curl (apt install curl) nebo python3 (apt install python3)." >&2; exit 1; }
mkdir -p tools/uv
if [ ! -x tools/uv/uv ]; then
    # připnutá verze a otisk SHA-256 (stejné jako UV_URL / UV_SHA256 v installer/fea_setup.py)
    mkdir -p tools/_download
    uvtgz=tools/_download/fea_uv.tar.gz
    curl -fsSL -o "$uvtgz" https://github.com/astral-sh/uv/releases/download/0.13.1/uv-x86_64-unknown-linux-gnu.tar.gz
    if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$uvtgz" | cut -d ' ' -f 1)
    else got=$(shasum -a 256 "$uvtgz" | cut -d ' ' -f 1); fi
    if [ "$got" != "eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220" ]; then
        rm -f "$uvtgz"; echo "CHYBA: otisk SHA-256 staženého uv nesouhlasí ($got) – instalace přerušena." >&2; exit 1
    fi
    tar -xzf "$uvtgz" -C tools/uv --strip-components=1
    rm -f "$uvtgz"
fi
# uv jen do složky programu: Python do tools/python, mezipaměť do tools/_download (instalátor ji na konci smaže),
# žádný spouštěč python3.11 v .local/bin domovské složky
export UV_PYTHON_INSTALL_DIR="$PWD/tools/python" UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_BIN=0
export UV_CACHE_DIR="${FEA_UV_CACHE_DIR:-$PWD/tools/_download/uv-cache}"
exec tools/uv/uv run --no-project --no-config --python 3.11 installer/fea_setup.py "$cmd" "$@"
