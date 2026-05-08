#!/bin/bash

# ── CONFIGURATION ────────────────────────────
POINTS=(100000 500000 1000000 5000000 10000000)
CLUSTERS=(10 20 30)
MAX_ITERATIONS=20
THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQ_SRC="$SCRIPT_DIR/main_sequential.cpp"
PAR_SRC="$SCRIPT_DIR/main_parallel.cpp"
OPT_SRC="$SCRIPT_DIR/main_optimized.cpp"
OCL_SRC="$SCRIPT_DIR/main_opencl.cpp"
KERNEL_SRC="$SCRIPT_DIR/kmeans_gpu.cl"

REPORT_FILE="$SCRIPT_DIR/benchmark_report.txt"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

compile_version() {
    local src="$1" bin="$2" p="$3" c="$4" iters="$5" mode="$6"
    
    # Use an array for flags to handle spaces in paths correctly
    local flags=("-O3" "-DNUM_POINTS=$p" "-DNUM_CLUSTERS=$c" "-DMAX_ITERATIONS=$iters")
    
    if [[ "$mode" == "opencl" ]]; then
        # When using arrays, we just need to wrap the macro value in quotes
        # The shell will pass it as a single argument to G++
        flags+=("-DKERNEL_FILE=\"$KERNEL_SRC\"")
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            flags+=("-framework" "OpenCL")
        else
            flags+=("-lOpenCL")
        fi
    else
        flags+=("-fopenmp")
    fi

    # Source file goes before flags/libraries for correct linking order
    g++ "$src" "${flags[@]}" -o "$bin"
    return $?
}

run_and_parse() {
    local bin="$1" threads="$2"
    if [[ ! -f "$bin" ]]; then echo "err"; return; fi
    
    local out
    out=$(OMP_NUM_THREADS="$threads" timeout 60s "$bin" 2>&1)
    
    # IMPROVED REGEX: 
    # 1. Look for the line containing "total time"
    # 2. Extract only the numbers/decimals
    # 3. Use 'tail -n 1' to ensure we get the LAST number on that line (the seconds)
    local t=$(echo "$out" | grep -i "total time" | grep -oP '[0-9.]+' | tail -n 1)
    
    if [[ -z "$t" ]]; then echo "err"; else echo "$t"; fi
}

echo -e "${BOLD}${CYAN}K-Means Multi-Version Benchmark${RESET}"
echo "------------------------------------------------"

declare -a RES_P RES_C RES_S RES_PR RES_O RES_G

for P in "${POINTS[@]}"; do
    for C in "${CLUSTERS[@]}"; do
        echo -ne "${YELLOW}P=$P C=$C... ${RESET}"
        
        compile_version "$SEQ_SRC" "./_b_s" "$P" "$C" "$MAX_ITERATIONS" "seq"
        compile_version "$PAR_SRC" "./_b_p" "$P" "$C" "$MAX_ITERATIONS" "par"
        compile_version "$OPT_SRC" "./_b_o" "$P" "$C" "$MAX_ITERATIONS" "opt"
        compile_version "$OCL_SRC" "./_b_g" "$P" "$C" "$MAX_ITERATIONS" "opencl"

        ST=$(run_and_parse "./_b_s" 1)
        PR=$(run_and_parse "./_b_p" "$THREADS")
        OT=$(run_and_parse "./_b_o" "$THREADS")
        GT=$(run_and_parse "./_b_g" 1)

        RES_P+=("$P"); RES_C+=("$C"); RES_S+=("$ST"); RES_PR+=("$PR"); RES_O+=("$OT"); RES_G+=("$GT")
        echo -e "${GREEN}Done${RESET}"
    done
done

# ... (rest of your table generation logic remains the same)

generate_table() {
    local clr=$1
    local B=$BOLD; local C=$CYAN; local R=$RESET; [[ "$clr" == "false" ]] && { B=""; C=""; R=""; }

    echo -e "${B}${C}========================================================================================${R}"
    printf "${B}%-10s %-3s %-10s %-10s %-10s %-10s %8s %8s %8s${R}\n" \
           "Points" "Cl" "Seq(s)" "Par(s)" "Opt(s)" "OCL(s)" "P-Spd" "O-Spd" "G-Spd"
    echo "----------------------------------------------------------------------------------------"
    for i in "${!RES_P[@]}"; do
        s="${RES_S[$i]}"; p="${RES_PR[$i]}"; o="${RES_O[$i]}"; g="${RES_G[$i]}"
        
        calc_spd() { awk -v s="$1" -v v="$2" 'BEGIN { if(v+0>0 && s+0>0) printf "%.2fx", s/v; else print "---" }'; }

        ps=$(calc_spd "$s" "$p"); os=$(calc_spd "$s" "$o"); gs=$(calc_spd "$s" "$g")

        # Format output strings
        [[ "$s" != "err" ]] && ss="${s}s" || ss="err"
        [[ "$p" != "err" ]] && ps_t="${p}s" || ps_t="err"
        [[ "$o" != "err" ]] && os_t="${o}s" || os_t="err"
        [[ "$g" != "err" ]] && gs_t="${g}s" || gs_t="err"

        printf "%-10s %-3s %-10s %-10s %-10s %-10s %8s %8s %8s\n" \
               "${RES_P[$i]}" "${RES_C[$i]}" "$ss" "$ps_t" "$os_t" "$gs_t" "$ps" "$os" "$gs"
    done
    echo -e "${B}${C}========================================================================================${R}"
}

generate_table "true"
generate_table "false" > "$REPORT_FILE"
echo -e "\nReport saved to: ${CYAN}$REPORT_FILE${RESET}"