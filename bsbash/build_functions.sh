#!/usr/bin/env bash
# Build functions used by build.sh and by the build.sh of each directory.
#
#   COMPILE_STATIC_LIB  <name> "<sources>" [extra flags]
#   COMPILE_EXECUTABLE  <name> "<sources>" [extra flags] [extra link flags]
#   ADD_SUBDIRECTORY    <dir>
#
# Objects go to $BUILD_DIR/<name>/, libraries to $LIB_DIR, executables to
# $BIN_DIR. Builds are incremental: an object is recompiled only when its
# source, one of the headers it includes (from the compiler's .d file) or the
# compiler flags changed.

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    COLOR_RED=$'\e[31m'; COLOR_GREEN=$'\e[32m'; COLOR_YELLOW=$'\e[33m'
    COLOR_BLUE=$'\e[34m'; COLOR_BOLD=$'\e[1m'; COLOR_RESET=$'\e[0m'
else
    COLOR_RED=""; COLOR_GREEN=""; COLOR_YELLOW=""; COLOR_BLUE=""; COLOR_BOLD=""; COLOR_RESET=""
fi

PRINT_INFO()  { [ "$QUIET" = "1" ] || echo "${COLOR_BLUE}--${COLOR_RESET} $*"; }
PRINT_OK()    { [ "$QUIET" = "1" ] || echo "${COLOR_GREEN}${COLOR_BOLD}OK${COLOR_RESET} $*"; }
PRINT_WARN()  { echo "${COLOR_YELLOW}warning:${COLOR_RESET} $*" >&2; }
PRINT_ERROR() { echo "${COLOR_RED}${COLOR_BOLD}error:${COLOR_RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# True if $1 (object) must be rebuilt from $2 (source) given $3 (dep file).
_NEEDS_REBUILD() {
    local obj=$1 src=$2 dep=$3
    [ -f "$obj" ] || return 0
    [ -f "$dep" ] || return 0
    [ "$src" -nt "$obj" ] && return 0
    local header
    # The .d file lists "obj: src header1 header2 ..." with line continuations.
    for header in $(sed -e 's/^[^:]*://' -e 's/\\$//' "$dep"); do
        case "$header" in *:) continue ;; esac
        [ -f "$header" ] && [ "$header" -nt "$obj" ] && return 0
    done
    return 1
}

# Removes the objects of a target when its flags changed since the last build.
_CHECK_FLAGS() {
    local dir=$1 flags=$2
    local stamp="$dir/.flags"
    if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$flags" ]; then
        rm -f "$dir"/*.o "$dir"/*.d
        printf '%s' "$flags" > "$stamp"
    fi
}

# Compiles a list of sources into $obj_dir, $JOBS at a time, and sets the
# OBJECTS array with the resulting object files.
_COMPILE_OBJECTS() {
    local obj_dir=$1 flags=$2
    shift 2
    local sources=("$@")
    local total=${#sources[@]} index=0
    OBJECTS=()

    mkdir -p "$obj_dir"
    _CHECK_FLAGS "$obj_dir" "$CXX $flags"
    rm -f "$obj_dir"/*.failed

    local src
    for src in "${sources[@]}"; do
        index=$((index + 1))
        local base
        base=$(echo "$src" | sed -e 's|^\./||' -e 's|[/.]|_|g')
        local obj="$obj_dir/$base.o" dep="$obj_dir/$base.d"
        OBJECTS+=("$obj")

        _NEEDS_REBUILD "$obj" "$src" "$dep" || continue

        local cmd="$CXX $flags -MMD -MP -MF $dep -c $src -o $obj"
        [ "$QUIET" = "1" ] || printf '   [%2d/%2d] %s\n' "$index" "$total" "$src"
        [ "$VERBOSE" = "1" ] && echo "         $cmd"

        # A failed compilation leaves a marker; its object is removed so the
        # next build retries it.
        ( eval "$cmd" || { rm -f "$obj"; echo "$src" > "$obj_dir/$base.failed"; } ) &

        # Keep at most $JOBS compilations running.
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
            wait -n 2>/dev/null || true
        done
    done
    wait

    local marker failed=0
    for marker in "$obj_dir"/*.failed; do
        [ -f "$marker" ] || continue
        PRINT_ERROR "compilation failed: $(cat "$marker")"
        failed=1
    done
    return $failed
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

# COMPILE_STATIC_LIB <name> "<sources>" [extra flags]
COMPILE_STATIC_LIB() {
    local name=$1 sources=$2 extra=$3
    local flags="$CXXFLAGS $extra"
    local lib="$LIB_DIR/lib$name.a"

    PRINT_INFO "${COLOR_BOLD}library $name${COLOR_RESET}"
    # shellcheck disable=SC2086
    _COMPILE_OBJECTS "$BUILD_DIR/$name" "$flags" $sources || return 1

    local relink=0 obj
    [ -f "$lib" ] || relink=1
    for obj in "${OBJECTS[@]}"; do [ "$obj" -nt "$lib" ] && relink=1; done
    if [ "$relink" = "1" ]; then
        mkdir -p "$LIB_DIR"
        rm -f "$lib"
        [ "$VERBOSE" = "1" ] && echo "         ar rcs $lib ${OBJECTS[*]}"
        ar rcs "$lib" "${OBJECTS[@]}" || { PRINT_ERROR "ar failed for $lib"; return 1; }
    fi
    PRINT_OK "$lib"
}

# COMPILE_EXECUTABLE <name> "<sources>" [extra flags] [extra link flags]
COMPILE_EXECUTABLE() {
    local name=$1 sources=$2 extra=$3 link=$4
    local flags="$CXXFLAGS $extra"
    local exe="$BIN_DIR/$name"

    PRINT_INFO "${COLOR_BOLD}executable $name${COLOR_RESET}"
    # shellcheck disable=SC2086
    _COMPILE_OBJECTS "$BUILD_DIR/$name" "$flags" $sources || return 1

    local relink=0 obj item
    [ -f "$exe" ] || relink=1
    for obj in "${OBJECTS[@]}"; do [ "$obj" -nt "$exe" ] && relink=1; done
    # Relink when a static library given as a path is newer than the executable.
    for item in $link; do
        case "$item" in *.a) [ -f "$item" ] && [ "$item" -nt "$exe" ] && relink=1 ;; esac
    done

    if [ "$relink" = "1" ]; then
        mkdir -p "$BIN_DIR"
        local cmd="$CXX ${OBJECTS[*]} -o $exe $LDFLAGS $link"
        [ "$VERBOSE" = "1" ] && echo "         $cmd"
        eval "$cmd" || { PRINT_ERROR "link failed for $exe"; return 1; }
    fi
    PRINT_OK "$exe"
}

# ADD_SUBDIRECTORY <dir>: runs <dir>/build.sh with the current settings.
ADD_SUBDIRECTORY() {
    local dir=$1
    if [ ! -f "$dir/build.sh" ]; then
        PRINT_ERROR "$dir/build.sh not found"
        return 1
    fi
    # shellcheck disable=SC1091
    ( CURRENT_DIR="$dir"; cd "$ROOT_DIR" && source "$dir/build.sh" ) || return 1
}
