set -euo pipefail

echo "yes" | ./build_visit3_4_2 --required --optional --mesagl --mpich --no-moab --no-visit --thirdparty-path /home/visit/third-party --makeflags -j4; python3 build_visit_docker_cleanup.py
