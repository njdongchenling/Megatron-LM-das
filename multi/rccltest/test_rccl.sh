#!/bin/bash
# rccl-tests 单测示例
#
# 单机单进程（1 进程驱动 N 卡，走 -g）:
#     bash test_rccl.sh single
#     GPUS_PER_NODE=4 MAXBYTES=8g bash test_rccl.sh single
#
# 多机多进程（必须使用编译了 MPI 的 rccl-tests，每卡 1 进程，不带 -g）:
#     # 先把节点写入 hostfile，一行一个节点名/IP（不用写 slots，脚本会自动补）
#     bash test_rccl.sh multi ./hostfile
#     SSH_PORT=11451 bash test_rccl.sh multi ./hostfile
#     OP=all_gather_perf bash test_rccl.sh multi ./hostfile
#
# 常用环境变量:
#     RCCL_TESTS   rccl-tests 的 build 目录，默认 /opt/rccl-test/build
#     OP           测试算子，默认 all_reduce_perf
#     GPUS_PER_NODE 每节点卡数，默认 8
#     IFNAME       指定可用网卡（-x NCCL_SOCKET_IFNAME），默认取 env.sh 里的值
#     SSH_PORT     mpirun 拉起远端进程用的 ssh 端口（--mca plm_rsh_args '-p N'），
#                  容器内 sshd 常不是 22，跨机必须按实际端口指定；留空则用默认 22
#     DTK_ENV      dtk 的 env.sh 路径，留空则不 source

set -o pipefail

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))

# Those variables need to modify
RCCL_TESTS=${RCCL_TESTS:-/opt/rccl-test/build}                # rccl-tests 编译产物目录
DTK_ENV=${DTK_ENV:-""}                                        # where env.sh of dtk
NCCL_ENV=${NCCL_ENV:-${MEGATRON_PATH}/requirements/env.sh}    # NCCL/RCCL 环境变量

# Those variables no need to modify
OP=${OP:-all_reduce_perf}       # all_reduce_perf / all_gather_perf / reduce_scatter_perf / alltoall_perf ...
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MINBYTES=${MINBYTES:-8}         # -b
MAXBYTES=${MAXBYTES:-4g}        # -e
STEPFACTOR=${STEPFACTOR:-2}     # -f
ITERS=${ITERS:-20}              # -n
WARMUP=${WARMUP:-5}             # -w
CHECK=${CHECK:-1}               # -c，1=校验结果正确性，压性能时可设 0
SSH_PORT=${SSH_PORT:-""}        # mpirun 拉起远端进程的 ssh 端口，留空用默认 22

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

MODE=${1:-single}
shift 2>/dev/null

log_info()  { echo -e "${GREEN}[INFO ]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

prepare_env() {
    [ -n "${DTK_ENV}" ] && { log_info "source ${DTK_ENV}"; source ${DTK_ENV}; }
    if [ -f "${NCCL_ENV}" ]; then
        log_info "source ${NCCL_ENV}"
        source ${NCCL_ENV}
    else
        log_error "NCCL_ENV not found: ${NCCL_ENV}"
        exit 1
    fi
    # 允许命令行覆盖 env.sh 里的网卡
    if [ -n "${IFNAME}" ]; then
        export NCCL_SOCKET_IFNAME=${IFNAME}
        export GLOO_SOCKET_IFNAME=${IFNAME}
    fi
    log_info "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}  NCCL_IB_HCA=${NCCL_IB_HCA}"

    BIN=${RCCL_TESTS}/${OP}
    if [ ! -x "${BIN}" ]; then
        log_error "rccl-tests binary not found: ${BIN}"
        log_error "请确认 RCCL_TESTS=${RCCL_TESTS} 指向 rccl-tests 的 build 目录（多机需带 MPI 编译: make MPI=1 MPI_HOME=... HIP_HOME=...）"
        exit 1
    fi
}

# 单机单进程：./all_reduce_perf -b 8 -e 1g -f 2 -g 4
run_single() {
    local ngpus=${GPUS_PER_NODE}
    # 支持 test_rccl.sh single -g 4 这样直接透传参数
    log_info "单机单进程: ${BIN} -b ${MINBYTES} -e ${MAXBYTES} -f ${STEPFACTOR} -g ${ngpus} $*"
    ${BIN} -b ${MINBYTES} \
           -e ${MAXBYTES} \
           -f ${STEPFACTOR} \
           -g ${ngpus} \
           -n ${ITERS} \
           -w ${WARMUP} \
           -c ${CHECK} \
           "$@"
}

# 多机多进程：mpirun --allow-run-as-root -np 16 --hostfile <file> -x LD_LIBRARY_PATH -x ROCM_PATH ./all_reduce_perf -b 8 -e 1g -f 2
run_multi() {
    local hostfile=$1
    shift 2>/dev/null

    if [ -z "${hostfile}" ] || [ ! -f "${hostfile}" ]; then
        log_error "用法: bash test_rccl.sh multi <hostfile>   # hostfile 一行一个节点名/IP"
        exit 1
    fi

    # 生成带 slots=N 的 mpi hostfile。
    # 注意: Open MPI 5 (PRRTE) 下 "-H <ip>:8" 不被接受（IP 形式解析不到本地节点，
    # 即使 -np 1 也会报 "not enough slots"），必须用 --hostfile + slots=N。
    local mpi_hostfile nodenum np
    mpi_hostfile=$(mktemp /tmp/rccl_hostfile.XXXXXX)
    awk -v n=${GPUS_PER_NODE} 'NF{print $1" slots="n}' ${hostfile} | sort -u > ${mpi_hostfile}
    nodenum=$(wc -l < ${mpi_hostfile})
    np=$((nodenum * GPUS_PER_NODE))

    if [ ${nodenum} -eq 0 ]; then
        log_error "hostfile 为空: ${hostfile}"
        rm -f ${mpi_hostfile}
        exit 1
    fi

    log_info "多机多进程: ${nodenum} 节点 / ${np} 进程 (每卡 1 进程，不带 -g)"
    log_info "hosts: $(paste -sd' ' ${mpi_hostfile})"

    # mpirun 默认走 22 端口 ssh 到远端节点拉起进程。容器里的 sshd 常映射到别的
    # 端口，此时必须用 plm_rsh_args 传 -p，否则跨机会卡在连接超时/被拒。
    local rsh_args=()
    if [ -n "${SSH_PORT}" ]; then
        rsh_args=(--mca plm_rsh_args "-p ${SSH_PORT}")
        log_info "ssh port: ${SSH_PORT}"
    fi

    mpirun --allow-run-as-root \
           -np ${np} \
           --hostfile ${mpi_hostfile} \
           --bind-to none \
           --mca plm_rsh_no_tree_spawn 1 \
           "${rsh_args[@]}" \
           -x LD_LIBRARY_PATH \
           -x ROCM_PATH \
           -x PATH \
           -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
           -x NCCL_IB_HCA \
           -x NCCL_ALGO \
           -x NCCL_NET_GDR_LEVEL \
           -x NCCL_NET_GDR_READ \
           -x RCCL_SDMA_COPY_ENABLE \
           -x NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
           ${BIN} -b ${MINBYTES} \
                  -e ${MAXBYTES} \
                  -f ${STEPFACTOR} \
                  -n ${ITERS} \
                  -w ${WARMUP} \
                  -c ${CHECK} \
                  "$@"
    local ret=$?
    rm -f ${mpi_hostfile}
    return ${ret}
}

prepare_env

case "${MODE}" in
    single) run_single "$@" ;;
    multi)  run_multi  "$@" ;;
    *)
        log_error "未知模式: ${MODE}"
        echo "用法: bash test_rccl.sh {single|multi <hostfile>} [额外的 rccl-tests 参数]"
        exit 1
        ;;
esac

ret=$?
if [ ${ret} -eq 0 ]; then
    log_info "rccl test done, 关注输出末尾的 'Avg bus bandwidth' 与 '#wrong' 是否为 0"
else
    log_error "rccl test failed, exit code=${ret}"
fi
exit ${ret}
