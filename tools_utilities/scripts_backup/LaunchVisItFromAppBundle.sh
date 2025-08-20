set -euo pipefail

#!/bin/sh
exec $(dirname $0)/../Resources/bin/visit $* &
