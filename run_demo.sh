#!/usr/bin/env bash
set -euo pipefail

# ===[ h4x0r-style banner ]====================================================
# (color only when stdout is a TTY)
if [ -t 1 ]; then
  G=$'\e[1;32m'   # green bold
  C=$'\e[1;36m'   # cyan  bold
  N=$'\e[0m'      # reset
else
  G=''; C=''; N=''
fi

# 색을 켜고 배너를 찍은 뒤 리셋 (sed 안 써도 됨)
printf "%s" "$G"
cat <<'ASCII'
============================================================
                                                                                                                                                            
@@@@@@@@@@   @@@  @@@  @@@  @@@             @@@@@@@  @@@@@@@    @@@@@@   @@@  @@@   @@@@@@   @@@@@@@@   @@@@@@   @@@@@@@   @@@@@@@@@@   @@@@@@@@  @@@@@@@   
@@@@@@@@@@@  @@@  @@@@ @@@  @@@             @@@@@@@  @@@@@@@@  @@@@@@@@  @@@@ @@@  @@@@@@@   @@@@@@@@  @@@@@@@@  @@@@@@@@  @@@@@@@@@@@  @@@@@@@@  @@@@@@@@  
@@! @@! @@!  @@!  @@!@!@@@  @@!               @@!    @@!  @@@  @@!  @@@  @@!@!@@@  !@@       @@!       @@!  @@@  @@!  @@@  @@! @@! @@!  @@!       @@!  @@@  
!@! !@! !@!  !@!  !@!!@!@!  !@!               !@!    !@!  @!@  !@!  @!@  !@!!@!@!  !@!       !@!       !@!  @!@  !@!  @!@  !@! !@! !@!  !@!       !@!  @!@  
@!! !!@ @!@  !!@  @!@ !!@!  !!@  @!@!@!@!@    @!!    @!@!!@!   @!@!@!@!  @!@ !!@!  !!@@!!    @!!!:!    @!@  !@!  @!@!!@!   @!! !!@ @!@  @!!!:!    @!@!!@!   
!@!   ! !@!  !!!  !@!  !!!  !!!  !!!@!@!!!    !!!    !!@!@!    !!!@!!!!  !@!  !!!   !!@!!!   !!!!!:    !@!  !!!  !!@!@!    !@!   ! !@!  !!!!!:    !!@!@!    
!!:     !!:  !!:  !!:  !!!  !!:               !!:    !!: :!!   !!:  !!!  !!:  !!!       !:!  !!:       !!:  !!!  !!: :!!   !!:     !!:  !!:       !!: :!!   
:!:     :!:  :!:  :!:  !:!  :!:               :!:    :!:  !:!  :!:  !:!  :!:  !:!      !:!   :!:       :!:  !:!  :!:  !:!  :!:     :!:  :!:       :!:  !:!  
:::     ::    ::   ::   ::   ::                ::    ::   :::  ::   :::   ::   ::  :::: ::    ::       ::::: ::  ::   :::  :::     ::    :: ::::  ::   :::  
 :      :    :    ::    :   :                  :      :   : :   :   : :  ::    :   :: : :     :         : :  :    :   : :   :      :    : :: ::    :   : :  
                                                                                                                                                             
============================================================
ASCII
printf "%s" "$N"

# 사설 IP(Private IP) 출력 - macOS 및 Linux 호환
get_private_ips() {
  local IPS=""
  
  # macOS에서 시도
  if command -v ifconfig >/dev/null 2>&1; then
    IPS="$(ifconfig | grep -E 'inet [0-9]' | grep -v '127.0.0.1' | awk '{print $2}' | grep -E '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' | paste -sd ' ' - 2>/dev/null || true)"
  fi
  
  # Linux에서 시도 (hostname -I)
  if [[ -z "$IPS" ]] && command -v hostname >/dev/null 2>&1; then
    IPS="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' | paste -sd ' ' - 2>/dev/null || true)"
  fi
  
  # Linux에서 시도 (ip 명령어)
  if [[ -z "$IPS" ]] && command -v ip >/dev/null 2>&1; then
    IPS="$(ip -4 addr show scope global 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | grep -E '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' | paste -sd ' ' - 2>/dev/null || true)"
  fi
  
  echo "${IPS:-N/A}"
}

PRIVATE_IPS="$(get_private_ips)"
printf "%b[INFO] Private IP(s): %s%b\n" "${C}" "${PRIVATE_IPS:-N/A}" "${N}"

# mini-transformer: 빌드 & 실행 헬퍼 스크립트 (서버 유지 실행)
# 사용법:
#   ./run_demo.sh                 # 현재 디렉터리(프로젝트 루트)에서 실행
#   ./run_demo.sh /path/to/proj   # 지정한 프로젝트 루트에서 실행
# 환경 변수:
#   JOBS    : 병렬 빌드 작업 수 지정 (기본: nproc 또는 2)
#   PORT    : API 서버 포트(lsof 확인용, 기본: 18080)
#   RUN_ARGS: 바이너리에 전달할 추가 인자(기본: --serve)

PROJECT_ROOT="${1:-$(pwd)}"
PORT="${PORT:-18080}"
# 기본 점검 포트 목록: $PORT 와 8080(일반적으로 충돌 잦음)
PORTS="${PORTS:-$PORT 8080}"

# macOS에서는 nproc 대신 sysctl 사용
get_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null || echo 2
  else
    echo 2
  fi
}

JOBS_DEFAULT="$(get_cpu_count)"
JOBS="${JOBS:-$JOBS_DEFAULT}"
RUN_ARGS="${RUN_ARGS:---serve}"

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*"; }
error() { echo "[ERROR] $*"; exit 1; }

if [[ ! -d "$PROJECT_ROOT" ]]; then
  error "Project root not found: $PROJECT_ROOT"
fi
if [[ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]]; then
  error "CMakeLists.txt not found in $PROJECT_ROOT"
fi

cd "$PROJECT_ROOT"
info "PROJECT_ROOT: $(pwd)"
info "User: $(whoami), Host: $(hostname)"
info "Kernel: $(uname -srmo)"

# 의존성 확인(없어도 진행) - macOS/Linux 호환
check_dependencies() {
  local missing_deps=()
  
  # CMake 확인
  if ! command -v cmake >/dev/null 2>&1; then
    missing_deps+=("cmake")
  fi
  
  # C++ 컴파일러 확인
  if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    missing_deps+=("c++ compiler (clang++ or g++)")
  fi
  
  # Make 확인
  if ! command -v make >/dev/null 2>&1; then
    missing_deps+=("make")
  fi
  
  if [[ ${#missing_deps[@]} -gt 0 ]]; then
    warn "Missing dependencies: ${missing_deps[*]}"
    if [[ "$(uname)" == "Darwin" ]]; then
      warn "On macOS, install with: brew install cmake"
      warn "For Xcode command line tools: xcode-select --install"
    else
      warn "On Linux, install with: sudo apt install cmake build-essential"
    fi
  fi
}

check_dependencies

# 포트 점유 프로세스 종료(있다면) - macOS/Linux 호환
free_port() {
  local P="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    if [[ "$(uname)" == "Darwin" ]]; then
      warn "'lsof' not found. On macOS it should be available by default."
    else
      warn "'lsof' not found. To install on Debian/Ubuntu: sudo apt install -y lsof"
    fi
    return 0
  fi
  
  # macOS/Linux 호환 방식으로 PID 수집
  local PIDS_STR
  PIDS_STR="$(lsof -t -i TCP:"$P" -sTCP:LISTEN 2>/dev/null || true)"
  
  if [[ -n "$PIDS_STR" ]]; then
    # 문자열을 배열로 변환
    local PIDS=($PIDS_STR)
    info "Port :$P is in use by PIDs: ${PIDS[*]} — attempting to terminate..."
    
    if kill "${PIDS[@]}" 2>/dev/null; then
      sleep 1
    else
      warn "Graceful kill failed; escalating to SIGKILL"
      kill -9 "${PIDS[@]}" 2>/dev/null || warn "Failed to kill PIDs; you may need 'sudo' to terminate other users' processes."
      sleep 1
    fi
    
    # 남은 프로세스 확인
    local REMAIN_STR
    REMAIN_STR="$(lsof -t -i TCP:"$P" -sTCP:LISTEN 2>/dev/null || true)"
    
    if [[ -n "$REMAIN_STR" ]]; then
      local REMAIN=($REMAIN_STR)
      warn "Port :$P still appears busy (PIDs: ${REMAIN[*]}). Proceeding, but you may need to free the port manually."
    else
      info "Port :$P is now free."
    fi
  else
    info "Port :$P is free."
  fi
}

# CMake 구성(최초 1회)
mkdir -p build
if [[ ! -f build/CMakeCache.txt ]]; then
  info "Configuring CMake (Release)..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
else
  info "CMake cache found; skipping configure."
fi

# 빌드
info "병렬 빌드: $JOBS 작업"
cmake --build build -j "$JOBS"

# 실행 전 포트 비우기 (설정 포트 + 8080 등)
for p in $PORTS; do
  free_port "$p"
done

# 서버 유지 실행(기본: --serve)
BIN="./build/mini_transformer"
if [[ ! -x "$BIN" ]]; then
  error "Binary not found or not executable: $BIN"
fi
info "실행 인자: $RUN_ARGS"
info "서버를 실행합니다. 브라우저에서 http://localhost:$PORT/ 접속 (종료: Ctrl+C)"
set +e
"$BIN" $RUN_ARGS
RUN_STATUS=$?
set -e

# 실행 후 포트 상태 출력(참고용)
if command -v lsof >/dev/null 2>&1; then
  info "Checking port(s) with lsof..."
  for p in $PORTS; do
    echo "--- :$p ---"; lsof -i ":$p" 2>/dev/null || true
  done
else
  if [[ "$(uname)" == "Darwin" ]]; then
    warn "'lsof' should be available on macOS by default."
  else
    warn "'lsof' not found. To install on Debian/Ubuntu: sudo apt install -y lsof"
  fi
fi

if [[ "$(uname)" == "Darwin" ]]; then
  info "Tip: On macOS, use 'brew install' for additional packages."
else
  info "Tip: 방화벽(UFW)이 필요하다면 'sudo apt install -y ufw'로 설치하세요."
fi

exit "$RUN_STATUS"
