#!/bin/bash
# Diagnostic: how does this g++ turn on libstdc++ assertions, and which flag
# (if any) switches them off? Run inside WSL:  wsl bash scripts/diag_assertions.sh
set -u
D=$(mktemp -d); cd "$D"
cat > t.cpp <<'EOF'
#include <vector>
int main(){ std::vector<int> v(3); volatile long i = 100000000; v[i] = 1; return 0; }
EOF
printf '#undef _GLIBCXX_ASSERTIONS\n' > noassert.h

echo "== g++ version"
g++ --version | head -1

echo "== -D / -U / -f flags the driver passes to the compiler"
echo 'int main(){}' | g++ -x c++ - -o /dev/null -### 2>&1 | tr ' ' '\n' | tr -d '"' \
  | grep -E '^-(D|U|f)' | sort -u

echo "== libstdc++ header defaults"
grep -n -E "define +_GLIBCXX_(ASSERTIONS|HARDEN)" /usr/include/c++/*/x86_64-linux-gnu/bits/c++config.h 2>/dev/null | head -8

run() {
  g++ -std=gnu++17 "$@" t.cpp -o t 2>/dev/null || { echo "  [$*] compile failed"; return; }
  ./t 2>err >/dev/null; rc=$?
  echo "  [$*] rc=$rc $(head -c 50 err | tr '\n' ' ')"
}
echo "== experiments (rc 139 = SIGSEGV = goal; rc 134 = still aborting)"
run
run -U_GLIBCXX_ASSERTIONS
run -include noassert.h
run -fno-hardened
run -U_GLIBCXX_ASSERTIONS -U_GLIBCXX_HARDEN
run -D_GLIBCXX_HARDEN=0
cd /; rm -rf "$D"
